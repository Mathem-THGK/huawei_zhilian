import sys
import re
import ast
from collections import defaultdict, deque
import heapq
import time
import random
import math

# -----------------------------------------------------------------------------
# 赛题二：AI最短推理时延 - 完整解决方案
# 版本：混合策略版 (Greedy + Simulated Annealing)
# 核心思路：
# 1. 第一阶段 (Greedy)：快速采用贪心策略（选择每个算子内部总执行时间最短的Tiling），
#    以获得一个高质量的、可行的初始解。
# 2. 第二阶段 (SA)：将贪心解作为模拟退火的起点，在剩余时间内进行局部搜索，对解进行精调优化。
# -----------------------------------------------------------------------------

def parse_input_line(line):
    line = line.strip()
    match = re.match(r'(\w+)\((.*)\)', line, re.DOTALL)
    if not match: return None, None
    func_name, args_str = match.group(1), match.group(2)
    try:
        return func_name, ast.literal_eval(args_str)
    except (ValueError, SyntaxError):
        try:
            return func_name, ast.literal_eval(args_str.replace('[', '(').replace(']', ')'))
        except: 
            return None, None

class Scheduler:
    # Scheduler 类保持不变，它提供了正确的调度核心
    def __init__(self):
        self.soc_info = {}
        self.op_lib = {}
        self.available_tilings = defaultdict(list)

    def set_soc_info(self, cores, memory):
        core_id_counter = 0
        self.soc_info['core_map'] = {}
        temp_cores_by_type = defaultdict(list)
        for core_type, num in cores:
            for _ in range(num):
                self.soc_info['core_map'][core_id_counter] = core_type
                temp_cores_by_type[core_type].append(core_id_counter)
                core_id_counter += 1
        self.soc_info['cores_by_type'] = temp_cores_by_type
        self.soc_info['memory'] = dict(memory)

    def add_op_info(self, op_type, shape, tiling_id, sub_graph, run_detail, mem_detail):
        key = (op_type, shape, tiling_id)
        nodes = set(item[0] for item in run_detail)
        successors, in_degrees = defaultdict(list), defaultdict(int)
        for u, v in sub_graph:
            nodes.add(u); nodes.add(v)
            successors[u].append(v); in_degrees[v] += 1
        
        total_exec_time = sum(item[2] for item in run_detail)

        op_info = {
            'sub_graph_deps': sub_graph,
            'run_detail': {item[0]: {'core_type': item[1], 'exec_time': item[2]} for item in run_detail},
            'mem_detail': defaultdict(lambda: defaultdict(int)),
            'initial_nodes': sorted([n for n in nodes if in_degrees[n] == 0]),
            'terminal_nodes': sorted([n for n in nodes if not successors.get(n)]),
            'total_exec_time': total_exec_time
        }
        for nid, mtype, msize in mem_detail: op_info['mem_detail'][nid][mtype] += msize
        self.op_lib[key] = op_info
        if tiling_id not in self.available_tilings.get((op_type, shape), []):
            self.available_tilings[(op_type, shape)].append(tiling_id)
            self.available_tilings[(op_type, shape)].sort()

    def _build_full_dependency_graph(self, op_nodes_list, tiling_choices):
        self.nodes, self.successors, self.predecessors, self.predecessor_count, self.op_info_map = set(), defaultdict(list), defaultdict(list), defaultdict(int), {}
        self.op_nodes_list = op_nodes_list
        for op_id, op_type, shape in self.op_nodes_list:
            key = (op_type, shape, tiling_choices.get(op_id))
            if key not in self.op_lib: return False
            self.op_info_map[op_id] = self.op_lib[key]
            for sub_id in self.op_info_map[op_id]['run_detail']: self.nodes.add((op_id, sub_id))
        
        for op_id, info in self.op_info_map.items():
            for u, v in info['sub_graph_deps']:
                u_node, v_node = (op_id, u), (op_id, v)
                if u_node in self.nodes and v_node in self.nodes:
                    self.successors[u_node].append(v_node); self.predecessors[v_node].append(u_node); self.predecessor_count[v_node] += 1
        
        op_succs = defaultdict(list); [op_succs[u].append(v) for u, v in self.op_graph_info]
        for u_op, v_ops in op_succs.items():
            u_info = self.op_info_map.get(u_op)
            for v_op in v_ops:
                v_info = self.op_info_map.get(v_op)
                if u_info and v_info:
                    for u_sub in u_info['terminal_nodes']:
                        for v_sub in v_info['initial_nodes']:
                            u_node, v_node = (u_op, u_sub), (v_op, v_sub)
                            self.successors[u_node].append(v_node); self.predecessors[v_node].append(u_node); self.predecessor_count[v_node] += 1
        
        self.mem_succ_counts = {node: defaultdict(int) for node in self.nodes}
        for node, succs in self.successors.items():
            node_op_id, node_sub_id = node
            mem_used_by_node = self.op_info_map[node_op_id]['mem_detail'][node_sub_id]
            if not mem_used_by_node: continue
            
            for succ in succs:
                succ_op_id, succ_sub_id = succ
                mem_used_by_succ = self.op_info_map[succ_op_id]['mem_detail'][succ_sub_id]
                for mem_type in set(mem_used_by_node.keys()).intersection(set(mem_used_by_succ.keys())):
                    self.mem_succ_counts[node][mem_type] += 1
        return True

    def _calculate_critical_path_priority(self):
        self.priorities = defaultdict(int)
        q = deque([n for n in self.nodes if not self.successors.get(n)])
        visited = set(q)
        while q:
            u_node = q.popleft()
            op_id, sub_id = u_node
            exec_time = self.op_info_map[op_id]['run_detail'][sub_id]['exec_time']
            max_p = max((self.priorities.get(v, 0) for v in self.successors.get(u_node, [])), default=0)
            self.priorities[u_node] = exec_time + max_p
            for pred in self.predecessors.get(u_node, []):
                if pred not in visited:
                    all_succs_visited = all(succ_of_pred in visited for succ_of_pred in self.successors.get(pred, []))
                    if all_succs_visited:
                        visited.add(pred)
                        q.append(pred)

    def schedule_attempt(self, op_graph, op_nodes, tiling_choices):
        self.op_graph_info = op_graph
        if not self._build_full_dependency_graph(op_nodes, tiling_choices):
            return float('inf'), None

        self._calculate_critical_path_priority()
        
        schedule_result, current_time, core_busy_until, memory_in_use = [], 0, defaultdict(int), defaultdict(int)
        running_nodes, ready_queue, finished_count, node_finish_times = [], [], 0, {}
        node_mem_detail_cache = {node: self.op_info_map[node[0]]['mem_detail'][node[1]] for node in self.nodes}
        
        local_predecessor_count = self.predecessor_count.copy()

        for node in self.nodes:
            if local_predecessor_count.get(node, 0) == 0:
                heapq.heappush(ready_queue, (-self.priorities[node], node))
        
        while finished_count < len(self.nodes):
            nodes_to_reschedule = []
            
            while ready_queue:
                priority, node = heapq.heappop(ready_queue)
                op_id, sub_id = node
                detail = self.op_info_map[op_id]['run_detail'][sub_id]
                core_type, mem_needed = detail['core_type'], node_mem_detail_cache[node]
                
                available_cores = [cid for cid in self.soc_info['cores_by_type'].get(core_type, []) if core_busy_until.get(cid, 0) <= current_time]
                mem_available = all(memory_in_use.get(m, 0) + s <= self.soc_info['memory'][m] for m, s in mem_needed.items())

                if available_cores and mem_available:
                    core_id = min(available_cores)
                    finish_time = current_time + detail['exec_time']
                    
                    for mt, ms in mem_needed.items(): 
                        memory_in_use[mt] = memory_in_use.get(mt, 0) + ms

                    for pred in self.predecessors.get(node, []):
                        pred_op_id, pred_sub_id = pred
                        mem_used_by_pred = self.op_info_map[pred_op_id]['mem_detail'][pred_sub_id]
                        for mem_type in set(mem_used_by_pred.keys()).intersection(set(mem_needed.keys())):
                            self.mem_succ_counts[pred][mem_type] -= 1
                            if self.mem_succ_counts[pred][mem_type] == 0:
                                memory_in_use[mem_type] -= mem_used_by_pred[mem_type]
                    
                    core_busy_until[core_id] = finish_time
                    heapq.heappush(running_nodes, (finish_time, node, core_id))
                    schedule_result.append([op_id, tiling_choices[op_id], sub_id, current_time, core_id])
                else:
                    nodes_to_reschedule.append((priority, node))
            
            for item in nodes_to_reschedule: heapq.heappush(ready_queue, item)
            
            if not running_nodes:
                if ready_queue: return float('inf'), None
                else: break

            next_event_time = running_nodes[0][0]
            if not ready_queue or next_event_time > current_time:
                 current_time = next_event_time
            
            while running_nodes and running_nodes[0][0] <= current_time:
                finish_time, node, core_id = heapq.heappop(running_nodes)
                finished_count += 1
                node_finish_times[node] = finish_time
                
                mem_detail = node_mem_detail_cache[node]
                for mem_type, mem_size in mem_detail.items():
                    if self.mem_succ_counts[node].get(mem_type, 0) == 0:
                        memory_in_use[mem_type] -= mem_size
                
                for succ in self.successors.get(node, []):
                    local_predecessor_count[succ] -= 1
                    if local_predecessor_count[succ] == 0:
                        heapq.heappush(ready_queue, (-self.priorities[succ], succ))
        
        if finished_count == len(self.nodes):
            makespan = max(node_finish_times.values()) if node_finish_times else 0
            return makespan, sorted(schedule_result)
        else:
            return float('inf'), None

def main():
    start_of_main = time.time()
    
    scheduler = Scheduler()
    content = sys.stdin.read()
    lines = content.strip().split('\n')
    
    op_info_lines, get_inference_line, set_soc_line = [], None, None
    for line in lines:
        if "AddOpInfo" in line: op_info_lines.append(line)
        elif "GetInferenceScheResult" in line: get_inference_line = line
        elif "SetSocInfo" in line: set_soc_line = line
    
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

            # --- STAGE 1, Plan A: 使用贪心策略 ---
            sys.stderr.write("[Hybrid] Stage 1: 尝试使用贪心策略生成初始解...\n")
            
            greedy_choice = {}
            for op_id, (op_type, shape) in op_map.items():
                available = scheduler.available_tilings.get((op_type, shape), [])
                if not available:
                    sys.stderr.write(f"[错误] 算子 {op_id} 没有任何可用的Tiling策略。\n")
                    print("[]")
                    return

                best_tiling_id = -1
                min_exec_time = float('inf')

                for tiling_id in available:
                    key = (op_type, shape, tiling_id)
                    op_info = scheduler.op_lib.get(key)
                    if op_info and op_info['total_exec_time'] < min_exec_time:
                        min_exec_time = op_info['total_exec_time']
                        best_tiling_id = tiling_id
                
                greedy_choice[op_id] = best_tiling_id if best_tiling_id != -1 else available[0]

            initial_energy, initial_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, greedy_choice)

            # --- STAGE 1, Plan B: 如果贪心失败，则启动随机搜索作为后备 ---
            if initial_energy == float('inf'):
                sys.stderr.write("[警告] 贪心解导致死锁，启动随机搜索作为后备计划...\n")
                
                attempts = 0
                max_attempts = 1000 # 随机搜索的尝试上限
                
                # 循环直到找到可行解或达到上限
                while initial_energy == float('inf') and attempts < max_attempts:
                    # 生成一个完全随机的Tiling组合
                    random_choice = {
                        op_id: random.choice(scheduler.available_tilings[op_map[op_id]])
                        for op_id in op_map if op_map[op_id] in scheduler.available_tilings and scheduler.available_tilings[op_map[op_id]]
                    }
                    initial_energy, initial_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, random_choice)
                    attempts += 1
                
                # 如果随机搜索也失败了，那就真的没办法了
                if initial_energy == float('inf'):
                    sys.stderr.write(f"[错误] 在{max_attempts}次随机尝试后，仍未能找到任何可行解。程序退出。\n")
                    print("[]")
                    return
                else:
                    # 如果随机搜索成功，将它的结果作为起点
                    greedy_choice = random_choice


            # --- STAGE 2: 以找到的可行解为起点，进行模拟退火精调 ---
            sys.stderr.write(f"[Hybrid] Stage 2: 以找到的可行解 (T={initial_energy:.0f}) 为起点进行模拟退火...\n")
            
            current_solution = greedy_choice
            current_energy = initial_energy
            current_schedule = initial_schedule

            best_solution = current_solution
            best_energy = current_energy
            best_schedule = current_schedule

            initial_temp = 1000.0
            final_temp = 1.0
            alpha = 0.99
            temp = initial_temp
            
            time_limit = 58.0 
            
            while temp > final_temp and (time.time() - start_of_main) < time_limit:
                neighbor_solution = current_solution.copy()
                op_to_change = random.choice(list(op_map.keys()))
                op_type, shape = op_map[op_to_change]
                available = scheduler.available_tilings.get((op_type, shape), [])
                if len(available) > 1:
                    new_tiling = random.choice([t for t in available if t != current_solution[op_to_change]])
                    neighbor_solution[op_to_change] = new_tiling
                else:
                    continue

                neighbor_energy, neighbor_schedule = scheduler.schedule_attempt(op_graph, op_nodes_info, neighbor_solution)
                
                if neighbor_energy == float('inf'):
                    continue

                energy_delta = neighbor_energy - current_energy
                if energy_delta < 0 or (random.random() < math.exp(-energy_delta / temp)):
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
                    current_schedule = neighbor_schedule
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = current_solution
                    best_schedule = current_schedule
                    sys.stderr.write(f"  [发现更优解] 新的T: {best_energy:.0f}, 温度: {temp:.2f}\n")
                
                temp *= alpha

            sys.stderr.write(f"\n[完成] 混合策略执行完毕。找到的最优时间为: {best_energy}\n")
            print(str(best_schedule).replace(" ", "") if best_schedule else "[]")

if __name__ == "__main__":
    main()
