import argparse
import xml.etree.ElementTree as ET
import re
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

console = Console()

def fix_collision_geom_names(input_xml, output_xml, extract_geoms_file=None, transparent_collision=True):
    """
    修复XML文件中碰撞几何体的名称，并可选择提取碰撞几何体信息
    
    规则：
    1. 已有名称的几何体保持不变
    2. 为碰撞几何体（没有contype="0" conaffinity="0"的几何体）添加名称
    3. 将碰撞几何体的名称设置为其所属身体的名称
    4. 如果同一身体下有多个碰撞几何体，添加编号后缀
    5. 可选：将碰撞几何体的rgba的alpha值设置为0（透明化）
    """
    
    print(Panel.fit(f"[bold cyan]处理文件:[/bold cyan] [yellow]{input_xml}[/yellow]", title="MuJoCo XML修复工具", border_style="cyan"))
    
    if transparent_collision:
        print(Panel.fit("[bold yellow]透明化设置:[/bold yellow] 碰撞几何体的rgba的alpha值将被设置为0（透明）", border_style="yellow"))
    
    # 解析XML文件
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("正在解析XML文件...", total=None)
        tree = ET.parse(input_xml)
        root = tree.getroot()
    
    # 用于存储已存在的名称，避免重复
    existing_names = set()
    
    # 用于收集碰撞几何体信息
    collision_geoms = []
    
    # 第一步：收集所有已有的几何体名称
    for geom in root.iter('geom'):
        if 'name' in geom.attrib:
            existing_names.add(geom.attrib['name'])
    
    # 第二步：处理所有几何体
    modified_count = 0
    transparent_count = 0
    
    with Progress() as progress:
        # 获取所有body的总数用于进度条
        bodies = list(root.iter('body'))
        task = progress.add_task("[cyan]处理碰撞几何体...", total=len(bodies))
        
        for body in bodies:
            body_name = body.get('name')
            if not body_name:
                # 跳过没有名称的身体
                progress.advance(task)
                continue
                
            # 为这个身体下的碰撞几何体计数
            collision_geom_counter = 0
            
            # 处理这个身体下的所有几何体
            for geom in body.findall('geom'):
                # 检查是否为碰撞几何体（没有contype="0" conaffinity="0"）
                contype = geom.get('contype', '')
                conaffinity = geom.get('conaffinity', '')
                
                if not (contype == '0' and conaffinity == '0'):
                    # 获取几何体类型
                    geom_type = geom.get('type', 'sphere')
                    
                    # 透明化处理
                    rgba_transparent = False
                    if transparent_collision:
                        rgba = geom.get('rgba', '')
                        if rgba:
                            # 解析rgba值
                            rgba_parts = rgba.split()
                            if len(rgba_parts) >= 4:
                                # 保存原始rgba值
                                original_rgba = rgba
                                # 设置alpha为0
                                rgba_parts[3] = '0'
                                new_rgba = ' '.join(rgba_parts)
                                geom.set('rgba', new_rgba)
                                rgba_transparent = True
                                transparent_count += 1
                    
                    # 如果几何体已有名称，直接记录信息
                    if 'name' in geom.attrib:
                        geom_name = geom.attrib['name']
                        collision_geoms.append({
                            'name': geom_name,
                            'type': geom_type,
                            'body': body_name,
                            'mesh': geom.get('mesh'),
                            'size': geom.get('size'),
                            'fromto': geom.get('fromto'),
                            'rgba_transparent': rgba_transparent,
                            'pre_existing': True
                        })
                        continue
                        
                    # 标记这个身体包含碰撞体
                    collision_geom_counter += 1
                    
                    # 生成基础名称（使用身体名称）
                    base_name = body_name
                    
                    # 如果基础名称已存在，添加编号
                    candidate_name = base_name
                    if collision_geom_counter > 1 or candidate_name in existing_names:
                        counter = 1
                        while candidate_name in existing_names:
                            candidate_name = f"{base_name}_{counter}"
                            counter += 1
                    
                    # 设置几何体名称
                    geom.set('name', candidate_name)
                    existing_names.add(candidate_name)
                    modified_count += 1
                    
                    # 记录碰撞几何体信息
                    collision_geoms.append({
                        'name': candidate_name,
                        'type': geom_type,
                        'body': body_name,
                        'mesh': geom.get('mesh'),
                        'size': geom.get('size'),
                        'fromto': geom.get('fromto'),
                        'rgba_transparent': rgba_transparent,
                        'pre_existing': False
                    })
            
            progress.advance(task)
    
    # 将XML写入字符串
    xml_str = ET.tostring(root, encoding='unicode', method='xml')
    
    # 后处理：去除XML声明和多余空格
    xml_str = post_process_xml(xml_str)
    
    # 写入文件
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(xml_str)
    
    # 显示结果面板
    result_summary = f"[bold green]✓ 处理完成！[/bold green]\n"
    result_summary += f"[bold]输出文件:[/bold] [yellow]{output_xml}[/yellow]\n"
    result_summary += f"[bold]新增名称:[/bold] [cyan]{modified_count}[/cyan] 个碰撞几何体\n"
    result_summary += f"[bold]总计数量:[/bold] [magenta]{len(collision_geoms)}[/magenta] 个碰撞几何体"
    
    if transparent_collision:
        result_summary += f"\n[bold]透明化:[/bold] [yellow]{transparent_count}[/yellow] 个几何体已设为透明"
    
    print(Panel.fit(
        result_summary,
        title="结果摘要",
        border_style="green",
        padding=(1, 2)
    ))
    
    # 处理碰撞几何体信息
    if collision_geoms:
        # 统计类型分布
        type_counts = {}
        for geom_info in collision_geoms:
            geom_type = geom_info['type']
            type_counts[geom_type] = type_counts.get(geom_type, 0) + 1
        
        # 创建类型统计表格
        type_table = Table(title="几何体类型统计", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        type_table.add_column("类型", style="cyan", justify="center")
        type_table.add_column("数量", style="green", justify="center")
        type_table.add_column("占比", style="yellow", justify="center")
        
        for geom_type, count in sorted(type_counts.items()):
            percentage = f"{(count / len(collision_geoms) * 100):.1f}%"
            type_table.add_row(geom_type, str(count), percentage)
        
        print(type_table)
        
        # 创建几何体详情表格
        detail_table = Table(
            title=f"碰撞几何体详情 (共 {len(collision_geoms)} 个)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold blue",
            show_lines=True if len(collision_geoms) > 10 else False
        )
        
        detail_table.add_column("序号", style="dim", width=4, justify="right")
        detail_table.add_column("几何体名称", style="cyan", overflow="fold")
        detail_table.add_column("类型", style="green", justify="center", width=10)
        detail_table.add_column("所属身体", style="magenta", overflow="fold")
        detail_table.add_column("状态", style="yellow", justify="center", width=6)
        detail_table.add_column("详细信息", style="dim", overflow="fold")
        
        # 按几何体名称排序
        collision_geoms.sort(key=lambda x: x['name'])
        
        for i, geom_info in enumerate(collision_geoms, 1):
            status = "[green]已有[/green]" if geom_info['pre_existing'] else "[yellow]新增[/yellow]"
            
            # 构建详细信息字符串
            details = []
            if geom_info['mesh']:
                details.append(f"mesh={geom_info['mesh']}")
            if geom_info['size']:
                details.append(f"size={geom_info['size']}")
            if geom_info['fromto']:
                details.append(f"fromto={geom_info['fromto']}")
            details_str = ", ".join(details)
            
            detail_table.add_row(
                str(i),
                geom_info['name'],
                geom_info['type'],
                geom_info['body'],
                status,
                details_str
            )
        
        print(detail_table)
        
        # 如果需要保存到文件
        if extract_geoms_file:
            # 保存为文本格式
            with open(extract_geoms_file, 'w', encoding='utf-8') as f:
                f.write("碰撞几何体列表\n")
                f.write("=" * 60 + "\n")
                f.write(f"总数量: {len(collision_geoms)}\n\n")
                
                f.write("类型统计:\n")
                for geom_type, count in sorted(type_counts.items()):
                    f.write(f"  {geom_type}: {count}个\n")
                
                f.write("\n详细列表:\n")
                for i, geom_info in enumerate(collision_geoms, 1):
                    status = "已有" if geom_info['pre_existing'] else "新增"
                    mesh_info = f", mesh={geom_info['mesh']}" if geom_info['mesh'] else ""
                    size_info = f", size={geom_info['size']}" if geom_info['size'] else ""
                    fromto_info = f", fromto={geom_info['fromto']}" if geom_info['fromto'] else ""
                    
                    f.write(f"{i:3d}. {geom_info['name']:30s} [类型: {geom_info['type']:10s}, "
                            f"所属body: {geom_info['body']:20s}, 状态: {status}]"
                            f"{mesh_info}{size_info}{fromto_info}\n")
            
            # 同时保存为JSON格式
            json_file = extract_geoms_file.replace('.txt', '.json') if extract_geoms_file.endswith('.txt') else extract_geoms_file + '.json'
            try:
                import json
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'total_count': len(collision_geoms),
                        'type_counts': type_counts,
                        'geoms': collision_geoms
                    }, f, indent=2, ensure_ascii=False)
                
                print(Panel.fit(
                    f"[bold green]✓ 文件已保存！[/bold green]\n"
                    f"[bold]文本格式:[/bold] [yellow]{extract_geoms_file}[/yellow]\n"
                    f"[bold]JSON格式:[/bold] [cyan]{json_file}[/cyan]",
                    title="文件保存状态",
                    border_style="blue",
                    padding=(1, 2)
                ))
            except ImportError:
                print(Panel(
                    "[bold red]警告:[/bold red] 无法保存JSON格式，需要json模块支持\n"
                    f"文本格式已保存到: [yellow]{extract_geoms_file}[/yellow]",
                    border_style="red"
                ))
    else:
        print(Panel(
            "[bold yellow]⚠ 没有找到碰撞几何体[/bold yellow]\n"
            "文件中可能没有碰撞几何体或所有几何体都是可视化几何体",
            border_style="yellow"
        ))
    
    return output_xml, collision_geoms

def post_process_xml(xml_str):
    """对XML字符串进行后处理，去除XML声明和/>前的空格"""
    
    # 1. 去除XML声明行（如果有的话）
    lines = xml_str.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('<?xml') and stripped.endswith('?>'):
            continue
        processed_lines.append(line)
    
    xml_str = '\n'.join(processed_lines)
    
    # 2. 去除自闭合标签中/>前的空格
    xml_str = re.sub(r'\s+/>', '/>', xml_str)
    
    return xml_str.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='修复MuJoCo XML文件中碰撞几何体的名称，并可选择提取碰撞几何体信息',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=str, required=True, help="输入XML文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出XML文件路径")
    parser.add_argument("--extract-geoms", type=str, help="可选项：提取碰撞几何体信息到指定文件（txt格式）")
    parser.add_argument("--transparent", action="store_true", help="将碰撞几何体透明化")
    
    args = parser.parse_args()

    try:
        # 显示程序标题
        print(Panel.fit(
            "[bold cyan]MuJoCo XML 碰撞几何体修复工具[/bold cyan]\n"
            "为碰撞几何体添加名称并提取详细信息",
            border_style="cyan",
            padding=(1, 4)
        ))
        
        fix_collision_geom_names(
            args.input, 
            args.output, 
            args.extract_geoms,
            args.transparent
        )
    except FileNotFoundError:
        print(Panel(
            f"[bold red]错误: 找不到文件 {args.input}[/bold red]",
            border_style="red"
        ))
    except ET.ParseError as e:
        print(Panel(
            f"[bold red]XML解析失败: {e}[/bold red]",
            border_style="red"
        ))
    except Exception as e:
        print(Panel(
            f"[bold red]错误: {e}[/bold red]",
            border_style="red"
        ))
        import traceback
        console.print_exception()
