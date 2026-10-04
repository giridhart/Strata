import sys
import re
import tree_sitter_cpp
from tree_sitter import Parser, Language

def apply_edits(source_bytes, edits):
    edits.sort(key=lambda e: e[0], reverse=True)
    for start, end, repl in edits:
        source_bytes = source_bytes[:start] + repl + source_bytes[end:]
    return source_bytes

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 refactor_ast.py <file.dp.cpp>")
        return

    parser = Parser(Language(tree_sitter_cpp.language()))
    
    with open(sys.argv[1], 'rb') as f:
        source = f.read()

    # Pre-AST text replacements for simple token replacements that are safe globally
    source_str = source.decode('utf-8')
    source_str = source_str.replace('#include <dpct/dpct.hpp>', '#include <cstdint>\n#include <algorithm>')
    source_str = source_str.replace('__dpct_inline__', 'inline')
    source_str = re.sub(r'const dpct::err0 [a-zA-Z0-9_]* = 0;\n\s*', '', source_str)
    source_str = re.sub(r'dpct::has_capability_or_fail\([\s\S]*?\);\n\s*', '', source_str)
    
    
    
    source_str = re.sub(r'bool check_launch\(const char\* what\) \{[\s\S]*?\}', 'bool check_launch(const char* what) { return true; }', source_str)

    # 3D to 1D items
    source_str = source_str.replace('sycl::nd_item<3> item_ct1', 'sycl::nd_item<1> item')
    source_str = source_str.replace('item_ct1.get_group(2)', 'item.get_group(0)')
    source_str = source_str.replace('item_ct1.get_local_range(2)', 'item.get_local_range(0)')
    source_str = source_str.replace('item_ct1.get_local_id(2)', 'item.get_local_id(0)')
    source_str = source_str.replace('item_ct1.get_group_range(2)', 'item.get_group_range(0)')
    source_str = source_str.replace('item_ct1', 'item')
    
    # Subgroup builtins
    source_str = source_str.replace('sycl::ext::oneapi::this_work_item::get_nd_item<3>()', 'sycl::ext::oneapi::this_work_item::get_nd_item<1>()')
    source_str = source_str.replace('dpct::select_from_sub_group', 'sycl::select_from_group')
    source_str = source_str.replace('dpct::shift_sub_group_left', 'sycl::shift_group_left')
    source_str = source_str.replace('sycl::ext::oneapi::this_work_item::get_sub_group()', 'item.get_sub_group()')
    source_str = re.sub(r'q\.get_device\(\)\.get_info<sycl::info::device::sub_group_sizes>\(\)', r'q->get_device().get_info<sycl::info::device::sub_group_sizes>()', source_str)
    
    # Atomics
    source_str = source_str.replace('dpct::atomic_fetch_add<sycl::access::address_space::generic_space>', 'sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>')

    # Queues
    source_str = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)->', 'q->', source_str)
    source_str = re.sub(r'\(\s*stream \? \(sycl::queue\*\)stream : &dpct::get_default_queue\(\)\s*\)', 'q', source_str)
    source_str = source_str.replace('dpct::get_in_order_queue()', '([](){ static sycl::queue q; return q; }())')
    source_str = source_str.replace('dpct::get_default_queue()', '([](){ static sycl::queue q; return q; }())')
    source_str = source_str.replace('(*([](){ static sycl::queue q; return q; }()))->', '([](){ static sycl::queue q; return q; }())->')
    
    # Dimensions 3D to 1D regex for simple cases
    source_str = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\s*\*\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
                  r'sycl::nd_range<1>(sycl::range<1>((\1) * (\2)), sycl::range<1>(\3))', source_str)
    source_str = re.sub(r'sycl::nd_range<3>\(\s*sycl::range<3>\(1,\s*1,\s*(.*?)\),\s*sycl::range<3>\(1,\s*1,\s*(.*?)\)\)', 
                  r'sycl::nd_range<1>(sycl::range<1>(\1), sycl::range<1>(\2))', source_str)
    source_str = re.sub(r'sycl::range<3>\(1,\s*1,\s*(.*?)\)', r'sycl::range<1>(\1)', source_str)
    source_str = re.sub(r'sycl::nd_range<3>', r'sycl::nd_range<1>', source_str)
    source_str = re.sub(r'sycl::nd_item<3>', r'sycl::nd_item<1>', source_str)

    source = source_str.encode('utf-8')
    tree = parser.parse(source)
    edits = []

    def walk(node):
        # Strip DPCT_CHECK_ERROR
        
        if node.type == 'expression_statement':
            expr = node.children[0]
            if expr.type == 'call_expression':
                func_node = expr.child_by_field_name('function')
                if func_node and func_node.text in (b'sync_if_needed', b'finish'):
                    edits.append((node.start_byte, node.end_byte, b''))
                    return

        if node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node and func_node.text == b"DPCT_CHECK_ERROR":
                args_node = node.child_by_field_name('arguments')
                if args_node and args_node.child_count >= 3:
                    expr_node = args_node.children[1]
                    edits.append((node.start_byte, node.end_byte, expr_node.text))
        
        # Scoping stream
        
        if node.type == 'function_definition':
            decl = node.child_by_field_name('declarator')
            if decl and decl.type == 'function_declarator':
                func_name_node = decl.child_by_field_name('declarator')
                if func_name_node and func_name_node.text in (b'sync_if_needed', b'finish'):
                    edits.append((node.start_byte, node.end_byte, b''))
                    return # skip children

        if node.type == 'function_definition':
            decl = node.child_by_field_name('declarator')
            
            # Check for try block
            body = node.child_by_field_name('body')
            if body and body.type == 'try_statement':
                # Strip the try keyword and catch blocks
                # try_statement -> [ 'try', compound_statement, catch_clause+ ]
                compound = body.child_by_field_name('body')
                if compound:
                    edits.append((body.start_byte, compound.start_byte, b''))
                    edits.append((compound.end_byte, body.end_byte, b''))
                # Replace body with the inner compound statement for further processing
                body = compound
                
            if decl and decl.type == 'function_declarator':
                params = decl.child_by_field_name('parameters')
                if params and any(b'stream' in p.text for p in params.children):
                    if body and body.type == 'compound_statement':
                        # avoid double injection
                        body_text = body.text
                        if b"static sycl::queue default_queue;" not in body_text:
                            open_brace_end = body.start_byte + 1
                            injection = b"\n    static sycl::queue default_queue;\n    sycl::queue* q = stream ? static_cast<sycl::queue*>(stream) : &default_queue;"
                            edits.append((open_brace_end, open_brace_end, injection))

        for child in node.children: walk(child)

    walk(tree.root_node)

    new_source = apply_edits(source, edits)

    # 12. Subgroup size injection
    # For any parallel_for lambda with sycl::nd_item that uses subgroup, inject [[intel::reqd_sub_group_size(32)]]
    # We do a quick string replacement post-AST for the lambda signatures
    new_str = new_source.decode('utf-8')
    # If the file has 'select_from_group' or 'shift_group_left', we add it to ALL nd_item lambdas
    if 'sycl::select_from_group' in new_str or 'sycl::shift_group_left' in new_str:
        new_str = re.sub(r'\[=\]\(sycl::nd_item<1> (item|item_ct1)\)\s*\{', r'[=](sycl::nd_item<1> \1) [[intel::reqd_sub_group_size(32)]] {', new_str)
    
    # bf16 types
    new_str = new_str.replace('dpct::bfloat16', 'sycl::ext::oneapi::bfloat16')
    new_str = new_str.replace('dpct::half_t', 'sycl::half')


    new_str = new_str.replace("const dpct::err0 s = dpct::get_current_device().queues_wait_and_throw();", "")
    new_str = new_str.replace("dpct::dim3", "sycl::range<3>")
    with open(sys.argv[1], 'wb') as f:
        f.write(new_str.encode('utf-8'))

if __name__ == "__main__":
    main()
