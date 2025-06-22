import sys
import re
import ast
from collections import defaultdict, deque
import heapq
import time
import random

# -----------------------------------------------------------------------------
# 赛题二：AI最短推理时延 - 完整解决方案
# 版本：迭代优化最终版
# 特性：实现“尝试-失败-调整-重试”的框架，以解决硬死锁问题
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
        # 在添加时就记录可用的tiling
        self.available_tilings[(op_type, shape)].append(tiling_id)
        self.available_tilings[(op_type, shape)].sort()

    def _build_full_dependency_graph(self, tiling_choices):
        print("[进度] 正在构建全局依赖图...", file=sys.stderr)
        start_time = time.time()

        self.nodes = set()
        self.successors = defaultdict(list)
        self.predecessors = defaultdict(list)
        self.predecessor_count = defaultdict(int)
        self.op_info_map = {}

        for op_id, op_type, shape in self.op_nodes_list:
            chosen_tiling_id = tiling_choices[op_id]
            key = (op_type, shape, chosen_tiling_id)
            if key not in self.op_lib:
                raise ValueError(f"错误：尝试使用一个不存在的Tiling组合 {key}")
            
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
        
        end_time = time.time()
        print(f"[完成] 优先级计算完成。耗时: {end_time - start_time:.2f} 秒。", file=sys.stderr)

    def schedule_attempt(self, op_graph, op_nodes, tiling_choices):
        self.op_graph_info = op_graph
        self.op_nodes_list = op_nodes
        
        self._build_full_dependency_graph(tiling_choices)
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
                    for pred_node in self.predecessors[node]:
                        if pred_node not in node_finish_times: continue
                        pred_mem = node_mem_detail_cache[pred_node]
                        for mem_type in pred_mem:
                            if mem_type in mem_needed:
                                pass # Memory release logic needs careful implementation
                else:
                    nodes_to_reschedule.append((priority, node))
            
            for item in nodes_to_reschedule:
                heapq.heappush(ready_queue, item)

            if not running_nodes:
                if ready_queue:
                    if not scheduled_in_this_pass:
                        print("\n" + "="*50, file=sys.stderr)
                        print(f"[死锁] 在时间 {current_time} 调度失败，返回并尝试更换Tiling策略。", file=sys.stderr)
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
                if finished_count % 500 == 0 or finished_count == len(self.nodes):
                    print(f"  [调度中] 已完成 {finished_count}/{len(self.nodes)} 个节点... 当前模拟时间: {current_time}", file=sys.stderr)
                core_busy_until[core_id] = 0
                # Simplified memory release
                mem_detail = node_mem_detail_cache[node]
                for m_type, m_size in mem_detail.items():
                    memory_in_use[m_type] -= m_size
                for succ_node in self.successors[node]:
                    self.predecessor_count[succ_node] -= 1
                    if self.predecessor_count[succ_node] == 0:
                        heapq.heappush(ready_queue, (-self.priorities[succ_node], succ_node))

        loop_end_time = time.time()
        if finished_count == len(self.nodes):
            max_finish_time = max(node_finish_times.values()) if node_finish_times else 0
            print(f"[成功] 调度成功。耗时: {loop_end_time - loop_start_time:.2f} 秒。最终模拟时间: {max_finish_time}", file=sys.stderr)
            print(str(sorted(schedule_result)).replace(" ", ""))
            return True, max_finish_time
        else:
            deadlock_info = {'time': current_time, 'blocked_nodes': ready_queue[:5]}
            return False, deadlock_info

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
        _, initial_args = parse_input_line(get_inference_line)
        if initial_args:
            op_graph, op_nodes_info = initial_args
            
            # --- 迭代重试框架 ---
            op_ids = [op[0] for op in op_nodes_info]
            # 初始Tiling策略：全部选择第一个可用的
            tiling_choices = {op_id: scheduler.available_tilings[(op[1], op[2])][0] for op_id, op in zip(op_ids, op_nodes_info)}

            max_retries = 20 # 最多尝试20种不同的Tiling组合
            success = False
            
            for i in range(max_retries):
                print(f"\n--- [尝试第 {i+1}/{max_retries} 轮调度] ---", file=sys.stderr)
                
                is_success, info = scheduler.schedule_attempt(op_graph, op_nodes_info, tiling_choices)
                
                if is_success:
                    success = True
                    break
                else:
                    deadlock_info = info
                    print(f"--- [第 {i+1} 轮调度失败] 发生死锁，尝试更换Tiling策略... ---", file=sys.stderr)
                    
                    if not deadlock_info['blocked_nodes']:
                        print("死锁但没有阻塞的节点，无法继续优化，程序终止。", file=sys.stderr)
                        break
                    
                    # 策略：找到第一个被阻塞的算子，将其tiling切换到下一个
                    # (op_id, sub_id)
                    blocked_node_op_id = deadlock_info['blocked_nodes'][0][1][0]
                    
                    op_type, shape = next(item[1:] for item in op_nodes_info if item[0] == blocked_node_op_id)
                    available = scheduler.available_tilings[(op_type, shape)]
                    current_tiling = tiling_choices[blocked_node_op_id]
                    
                    try:
                        current_idx = available.index(current_tiling)
                        if current_idx + 1 < len(available):
                            new_tiling = available[current_idx + 1]
                            print(f"策略：将算子 {blocked_node_op_id} 的 Tiling 从 {current_tiling} 更换为 {new_tiling}", file=sys.stderr)
                            tiling_choices[blocked_node_op_id] = new_tiling
                        else:
                            print(f"算子 {blocked_node_op_id} 所有Tiling策略均已尝试，无法解决。重置并尝试其他算子。", file=sys.stderr)
                            tiling_choices[blocked_node_op_id] = available[0] # 重置
                            # 在实际比赛中，这里可以实现更复杂的逻辑，比如随机选择另一个算子进行调整
                    except ValueError:
                        pass
            
            if not success:
                print("[]") # 所有重试均失败

if __name__ == "__main__":
    main()