import sys
import re
import ast
from collections import defaultdict, deque
import heapq
import time

# -----------------------------------------------------------------------------
# 赛题二：AI最短推理时延 - 完整解决方案
# 版本：修复了最终输出时的ValueError
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
        except Exception as e:
            sys.stderr.write(f"解析行失败: {line}, 错误: {e}\n")
            return None, None

class Scheduler:
    def __init__(self):
        self.soc_info = {}
        self.op_lib = {}
        self.op_graph_info = None
        self.op_nodes_list = None

    def set_soc_info(self, cores, memory):
        self.soc_info['core_map'] = {}
        core_id_counter = 0
        all_core_ids = []
        for core_type, num in cores:
            for i in range(num):
                self.soc_info['core_map'][core_id_counter] = core_type
                all_core_ids.append(core_id_counter)
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

    def _build_full_dependency_graph(self):
        print("[进度] 正在构建全局依赖图...", file=sys.stderr)
        start_time = time.time()

        self.nodes = set()
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.predecessor_count = defaultdict(int)
        
        self.op_tiling_choice = {}
        self.op_info_map = {}

        available_tilings = defaultdict(list)
        for op_type, shape, tiling_id in self.op_lib.keys():
            available_tilings[(op_type, shape)].append(tiling_id)
        
        for op_id, op_type, shape in self.op_nodes_list:
            if (op_type, shape) not in available_tilings:
                raise ValueError(f"错误：计算图请求了一个未在AddOpInfo中定义的算子 (OpType={op_type}, Shape={shape})")
            
            chosen_tiling_id = sorted(available_tilings[(op_type, shape)])[0]
            self.op_tiling_choice[op_id] = chosen_tiling_id
            
            key = (op_type, shape, chosen_tiling_id)
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
        
        end_time = time.time()
        print(f"[完成] 图构建完成。耗时: {end_time - start_time:.2f} 秒。图中有 {len(self.nodes)} 个节点。", file=sys.stderr)

    def _calculate_critical_path_priority(self):
        print("[进度] 正在计算所有节点的关键路径优先级...", file=sys.stderr)
        start_time = time.time()

        self.priorities = defaultdict(int)
        
        q = deque([n for n in self.nodes if not self.successors[n]])
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
                    all_succ_visited = all(succ in visited_nodes for succ in self.successors[pred_node])
                    if all_succ_visited:
                        visited_nodes.add(pred_node)
                        q.append(pred_node)
        
        end_time = time.time()
        print(f"[完成] 优先级计算完成。耗时: {end_time - start_time:.2f} 秒。", file=sys.stderr)

    def schedule(self, op_graph, op_nodes):
        self.op_graph_info = op_graph
        self.op_nodes_list = op_nodes
        
        self._build_full_dependency_graph()
        self._calculate_critical_path_priority()
        
        print("[进度] 开始主调度循环...", file=sys.stderr)
        loop_start_time = time.time()

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

        mem_release_waits = defaultdict(lambda: defaultdict(int))
        node_mem_detail_cache = {}
        for u_node in self.nodes:
            u_op_id, u_sub_id = u_node
            u_info = self.op_info_map[u_op_id]
            u_mem_detail = u_info['mem_detail'][u_sub_id]
            node_mem_detail_cache[u_node] = u_mem_detail
            for succ_node in self.successors[u_node]:
                succ_op_id, succ_sub_id = succ_node
                succ_info = self.op_info_map[succ_op_id]
                succ_mem_detail = succ_info['mem_detail'][succ_sub_id]
                for mem_type in u_mem_detail:
                    if mem_type in succ_mem_detail:
                        mem_release_waits[u_node][mem_type] += 1

        while finished_count < len(self.nodes):
            nodes_to_reschedule = []
            while ready_queue:
                priority, node = heapq.heappop(ready_queue)
                op_id, sub_id = node
                info = self.op_info_map[op_id]
                detail = info['run_detail'][sub_id]
                core_type_needed = detail['core_type']
                mem_needed = node_mem_detail_cache[node]

                available_cores = [cid for cid in self.soc_info['cores_by_type'][core_type_needed] if core_busy_until[cid] <= current_time]
                
                mem_available = all(memory_in_use[m_type] + m_size <= self.soc_info['memory'][m_type] for m_type, m_size in mem_needed.items())

                if available_cores and mem_available:
                    core_id_to_assign = min(available_cores)
                    core_busy_until[core_id_to_assign] = current_time + detail['exec_time']
                    
                    for m_type, m_size in mem_needed.items():
                        memory_in_use[m_type] += m_size
                    
                    heapq.heappush(running_nodes, (current_time + detail['exec_time'], node, core_id_to_assign))
                    schedule_result.append([op_id, self.op_tiling_choice[op_id], sub_id, current_time, core_id_to_assign])

                    for pred_node in self.predecessors[node]:
                        pred_mem = node_mem_detail_cache[pred_node]
                        for mem_type in pred_mem:
                            if mem_type in mem_needed:
                                mem_release_waits[pred_node][mem_type] -= 1
                                if mem_release_waits[pred_node][mem_type] == 0:
                                    memory_in_use[mem_type] -= pred_mem[mem_type]
                else:
                    nodes_to_reschedule.append((priority, node))
            
            for item in nodes_to_reschedule:
                heapq.heappush(ready_queue, item)

            if not running_nodes:
                if ready_queue:
                    current_time += 1
                else:
                    break
            else:
                next_finish_time = running_nodes[0][0]
                current_time = max(current_time, next_finish_time)
            
            while running_nodes and running_nodes[0][0] <= current_time:
                finish_time, node, core_id = heapq.heappop(running_nodes)
                finished_count += 1
                node_finish_times[node] = finish_time
                if finished_count % 100 == 0 or finished_count == len(self.nodes):
                    print(f"  [调度中] 已完成 {finished_count}/{len(self.nodes)} 个节点... 当前模拟时间: {current_time}", file=sys.stderr)

                mem_detail = node_mem_detail_cache[node]
                for m_type, m_size in mem_detail.items():
                    if mem_release_waits[node].get(m_type, 0) == 0:
                        memory_in_use[m_type] -= m_size

                for succ_node in self.successors[node]:
                    self.predecessor_count[succ_node] -= 1
                    if self.predecessor_count[succ_node] == 0:
                        heapq.heappush(ready_queue, (-self.priorities[succ_node], succ_node))

        loop_end_time = time.time()
        if finished_count == len(self.nodes):
            # --- 【核心修改点】修正解包逻辑 ---
            # 正确的修正后代码
            max_finish_time = max(node_finish_times.values()) if node_finish_times else 0

            print(f"[完成] 调度循环完成。耗时: {loop_end_time - loop_start_time:.2f} 秒。最终模拟时间: {max_finish_time}", file=sys.stderr)
            print("[进度] 正在格式化并输出最终结果...", file=sys.stderr)

            print(str(sorted(schedule_result)).replace(" ", ""))
        else:
            print(f"[错误] 调度未完成！可能存在死锁。已完成 {finished_count}/{len(self.nodes)}", file=sys.stderr)
            print("[]")

def main():
    print("[启动] 开始解析输入数据...", file=sys.stderr)
    scheduler = Scheduler()
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
    
    print(f"[完成] 输入解析完成。共有 {len(op_info_lines)} 条算子信息。", file=sys.stderr)

    if set_soc_line:
        _, args = parse_input_line(set_soc_line)
        if args: scheduler.set_soc_info(*args)

    for op_line in op_info_lines:
        _, op_args = parse_input_line(op_line)
        if op_args: scheduler.add_op_info(*op_args)

    if get_inference_line:
        _, args = parse_input_line(get_inference_line)
        if args: scheduler.schedule(*args)

if __name__ == "__main__":
    main()