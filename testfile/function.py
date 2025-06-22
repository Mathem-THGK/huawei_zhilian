import sys
import re
import ast
from collections import defaultdict, deque
import heapq
import time
import random

# -----------------------------------------------------------------------------
# 赛题二：AI最短推理时延 - 完整解决方案
# 版本：最终提交版 (输出格式按要求修改)
# -----------------------------------------------------------------------------

def parse_input_line(line):
    """更安全地解析单行输入"""
    line = line.strip()
    match = re.match(r'(\w+)\((.*)\)', line, re.DOTALL)
    if not match:
        return None, None
    func_name = match.group(1)
    args_str = match.group(2)
    try:
        args = ast.literal_eval(args_str)
        return func_name, args
    except (ValueError, SyntaxError):
        args_str_compat = args_str.replace('[', '(').replace(']', ')')
        try:
            args = ast.literal_eval(args_str_compat)
            return func_name, args
        except Exception:
            return None, None

class Scheduler:
    def __init__(self):
        self.soc_info = {}
        self.op_lib = {}
        self.op_graph_info = None
        self.op_nodes_list = None
        self.available_tilings = defaultdict(list)

    def set_soc_info(self, cores, memory):
        self.soc_info['core_map'] = {}
        core_id_counter = 0
        for core_type, num in cores:
            for i in range(num):
                self.soc_info['core_map'][core_id_counter] = core_type
                core_id_counter += 1
        
        self.soc_info['cores_by_type'] = {core_type: [cid for cid, ctype in self.soc_info['core_map'].items() if ctype == core_type] for core_type, _ in cores}
        self.soc_info['memory'] = dict(memory)

    def add_op_info(self, op_type, shape, tiling_id, sub_graph, run_detail, mem_detail):
        key = (op_type, shape, tiling_id)
        
        nodes = set()
        in_degrees = defaultdict(int)
        successors = defaultdict(list)
        for u, v in sub_graph:
            nodes.add(u)
            nodes.add(v)
            in_degrees[v] += 1
            successors[u].append(v)
        
        all_node_ids = {item[0] for item in run_detail}
        nodes.update(all_node_ids)

        initial_nodes = sorted([n for n in nodes if in_degrees[n] == 0])
        terminal_nodes = sorted([n for n in nodes if not successors.get(n) and n in nodes])

        op_info = {
            'sub_graph_deps': sub_graph,
            'run_detail': {item[0]: {'core_type': item[1], 'exec_time': item[2]} for item in run_detail},
            'mem_detail': defaultdict(lambda: defaultdict(int)),
            'initial_nodes': initial_nodes,
            'terminal_nodes': terminal_nodes,
        }
        for node_id, mem_type, mem_size in mem_detail:
            op_info['mem_detail'][node_id][mem_type] += mem_size
        
        self.op_lib[key] = op_info
        if tiling_id not in self.available_tilings[(op_type, shape)]:
            self.available_tilings[(op_type, shape)].append(tiling_id)
            self.available_tilings[(op_type, shape)].sort()

    def _build_full_dependency_graph(self, tiling_choices):
        self.nodes = set()
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.predecessor_count = defaultdict(int)
        self.op_info_map = {}

        for op_id, op_type, shape in self.op_nodes_list:
            chosen_tiling_id = tiling_choices[op_id]
            key = (op_type, shape, chosen_tiling_id)
            if key not in self.op_lib:
                raise ValueError(f"Tiling combination not found: {key}")
            
            self.op_info_map[op_id] = self.op_lib[key]
            for sub_node_id in self.op_info_map[op_id]['run_detail']:
                 self.nodes.add((op_id, sub_node_id))

        for op_id, info in self.op_info_map.items():
            for u_sub, v_sub in info['sub_graph_deps']:
                u_node, v_node = (op_id, u_sub), (op_id, v_sub)
                if u_node in self.nodes and v_node in self.nodes:
                    self.successors[u_node].append(v_node)
                    self.predecessors[v_node].append(u_node)
                    self.predecessor_count[v_node] += 1
        
        temp_op_successors = defaultdict(list)
        for u_op, v_op in self.op_graph_info:
            temp_op_successors[u_op].append(v_op)

        for u_op, v_ops in temp_op_successors.items():
            u_info = self.op_info_map.get(u_op)
            if not u_info: continue
            for v_op in v_ops:
                v_info = self.op_info_map.get(v_op)
                if not v_info: continue
                for u_sub in u_info['terminal_nodes']:
                    for v_sub in v_info['initial_nodes']:
                        u_node, v_node = (u_op, u_sub), (v_op, v_sub)
                        self.successors[u_node].append(v_node)
                        self.predecessors[v_node].append(u_node)
                        self.predecessor_count[v_node] += 1

    def _calculate_critical_path_priority(self):
        self.priorities = defaultdict(int)
        
        q = deque([n for n in self.nodes if not self.successors.get(n)])
        visited_nodes = set(q)

        while q:
            u_node = q.popleft()
            op_id, sub_node_id = u_node
            info = self.op_info_map[op_id]
            exec_time = info['run_detail'][sub_node_id]['exec_time']
            max_succ_priority = 0
            if self.successors[u_node]:
                max_succ_priority = max(self.priorities.get(v_node, 0) for v_node in self.successors[u_node])
            self.priorities[u_node] = exec_time + max_succ_priority
            for pred_node in self.predecessors[u_node]:
                if pred_node not in visited_nodes:
                    if all(succ in visited_nodes for succ in self.successors[pred_node]):
                        visited_nodes.add(pred_node)
                        q.append(pred_node)

    def schedule_attempt(self, op_graph, op_nodes, tiling_choices):
        self.op_graph_info = op_graph
        self.op_nodes_list = op_nodes
        
        self._build_full_dependency_graph(tiling_choices)
        self._calculate_critical_path_priority()

        schedule_result = []
        current_time = 0
        core_busy_until = defaultdict(int)
        memory_in_use = defaultdict(int)
        running_nodes = []
        ready_queue = []
        
        for node in self.nodes:
            if self.predecessor_count[node] == 0:
                heapq.heappush(ready_queue, (-self.priorities[node], node))
        
        finished_count = 0
        node_finish_times = {}
        node_mem_detail_cache = {node: self.op_info_map[node[0]]['mem_detail'][node[1]] for node in self.nodes}

        while finished_count < len(self.nodes):
            nodes_to_reschedule = []
            scheduled_in_this_pass = False
            while ready_queue:
                priority, node = heapq.heappop(ready_queue)
                op_id, sub_id = node
                info = self.op_info_map[op_id]
                detail = info['run_detail'][sub_id]
                core_type_needed = detail['core_type']
                mem_needed = node_mem_detail_cache[node]

                available_cores = [cid for cid in self.soc_info['cores_by_type'].get(core_type_needed, []) if core_busy_until.get(cid, 0) <= current_time]
                mem_available = all(memory_in_use.get(m_type, 0) + m_size <= self.soc_info['memory'][m_type] for m_type, m_size in mem_needed.items())

                if available_cores and mem_available:
                    core_id_to_assign = min(available_cores)
                    core_busy_until[core_id_to_assign] = current_time + detail['exec_time']
                    for m_type, m_size in mem_needed.items():
                        memory_in_use[m_type] = memory_in_use.get(m_type, 0) + m_size
                    heapq.heappush(running_nodes, (current_time + detail['exec_time'], node, core_id_to_assign))
                    schedule_result.append([op_id, tiling_choices[op_id], sub_id, current_time, core_id_to_assign])
                    scheduled_in_this_pass = True
                    # Simplified memory release logic for now
                else:
                    nodes_to_reschedule.append((priority, node))
            
            for item in nodes_to_reschedule:
                heapq.heappush(ready_queue, item)

            if not running_nodes:
                if ready_queue:
                    if not scheduled_in_this_pass:
                        deadlock_info = {'time': current_time, 'blocked_nodes': ready_queue[:5]}
                        return False, deadlock_info
                else:
                    break
            
            if not running_nodes:
                 if ready_queue:
                     current_time += 1
                     continue
                 else:
                     break

            next_event_time = running_nodes[0][0]
            current_time = next_event_time
            
            while running_nodes and running_nodes[0][0] <= current_time:
                finish_time, node, core_id = heapq.heappop(running_nodes)
                finished_count += 1
                node_finish_times[node] = finish_time
                core_busy_until[core_id] = 0
                mem_detail = node_mem_detail_cache[node]
                for m_type, m_size in mem_detail.items():
                    memory_in_use[m_type] -= m_size
                for succ_node in self.successors[node]:
                    self.predecessor_count[succ_node] -= 1
                    if self.predecessor_count[succ_node] == 0:
                        heapq.heappush(ready_queue, (-self.priorities[succ_node], succ_node))

        if finished_count == len(self.nodes):
            # --- 【核心修改点】按您的示例格式进行输出 ---
            print(sorted(schedule_result))
            return True, None
        else:
            deadlock_info = {'time': current_time, 'blocked_nodes': ready_queue[:5]}
            return False, deadlock_info

def main():
    scheduler = Scheduler()
    # Reading all at once is faster for large inputs
    content = sys.stdin.read()
    lines = content.strip().split('\n')
    
    op_info_lines = []
    get_inference_line = None
    set_soc_line = None

    for line in lines:
        if "AddOpInfo" in line:
            op_info_lines.append(line)
        elif "GetInferenceScheResult" in line:
            get_inference_line = line
        elif "SetSocInfo" in line:
            set_soc_line = line
    
    if set_soc_line:
        _, args = parse_input_line(set_soc_line)
        if args: scheduler.set_soc_info(*args)

    for op_line in op_info_lines:
        _, op_args = parse_input_line(op_line)
        if op_args: scheduler.add_op_info(*op_args)

    if get_inference_line:
        _, initial_args = parse_input_line(get_inference_line)
        if initial_args:
            op_graph, op_nodes_info = initial_args
            op_map = {op[0]: (op[1], op[2]) for op in op_nodes_info}

            tiling_choices = {op_id: scheduler.available_tilings[op_map[op_id]][0] for op_id in op_map if op_map[op_id] in scheduler.available_tilings and scheduler.available_tilings[op_map[op_id]]}

            max_retries = 20
            success = False
            
            for i in range(max_retries):
                is_success, info = scheduler.schedule_attempt(op_graph, op_nodes_info, tiling_choices)
                
                if is_success:
                    success = True
                    break
                else:
                    deadlock_info = info
                    if not deadlock_info or not deadlock_info.get('blocked_nodes'):
                        break
                    
                    # Simple strategy: randomly pick a blocked op and change its tiling
                    try:
                        blocked_node_op_id = deadlock_info['blocked_nodes'][0][1][0]
                        
                        op_type, shape = op_map[blocked_node_op_id]
                        available = scheduler.available_tilings[(op_type, shape)]
                        
                        if len(available) > 1:
                            new_tiling = random.choice([t for t in available if t != tiling_choices[blocked_node_op_id]])
                            tiling_choices[blocked_node_op_id] = new_tiling
                        else:
                             # If no other choice, try changing another random op
                            op_to_change = random.choice(list(op_map.keys()))
                            if op_map.get(op_to_change) in scheduler.available_tilings:
                                available = scheduler.available_tilings[op_map[op_to_change]]
                                if available:
                                    tiling_choices[op_to_change] = random.choice(available)
                    except (IndexError, KeyError):
                        # Break if deadlock info is malformed or op not found
                        break

            if not success:
                print("[]")

if __name__ == "__main__":
    main()