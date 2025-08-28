#!/usr/bin/env python3
"""
验证并修复UILabels类缺失问题
"""
import re
from pathlib import Path
import shutil
from datetime import datetime

def check_uilabels_in_settings():
    """检查settings.py中是否存在UILabels类"""
    print("检查config/settings.py中的UILabels...")
    
    settings_file = Path("config/settings.py")
    if not settings_file.exists():
        print("  [ERROR] config/settings.py不存在")
        return False
    
    with open(settings_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否包含UILabels类定义
    has_uilabels_class = 'class UILabels:' in content or 'class UILabels(' in content
    has_uilabels_instance = 'ui_labels = UILabels()' in content
    
    print(f"  UILabels类定义: {'存在' if has_uilabels_class else '缺失'}")
    print(f"  ui_labels实例: {'存在' if has_uilabels_instance else '缺失'}")
    
    if not has_uilabels_class:
        print("  [PROBLEM] UILabels类定义缺失")
        return False
    
    return True

def add_uilabels_to_settings():
    """添加UILabels类到settings.py"""
    print("添加UILabels类到config/settings.py...")
    
    settings_file = Path("config/settings.py")
    if not settings_file.exists():
        print("  [ERROR] config/settings.py不存在")
        return False
    
    # 备份原文件
    backup_name = f"settings_backup_uilabels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    backup_path = settings_file.parent / backup_name
    shutil.copy2(settings_file, backup_path)
    print(f"  已备份到: {backup_path}")
    
    with open(settings_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # UILabels类定义
    uilabels_class = '''
@dataclass
class UILabels:
    """用戶界面標籤配置（向後相容）"""
    # 主窗口標籤
    window_title: str = "無人機群模擬器"
    menu_file: str = "檔案"
    menu_edit: str = "編輯"
    menu_view: str = "檢視"
    menu_help: str = "說明"
    
    # 按鈕標籤
    btn_start: str = "開始"
    btn_stop: str = "停止"
    btn_pause: str = "暫停"
    btn_reset: str = "重置"
    btn_load: str = "載入"
    btn_save: str = "儲存"
    
    # 狀態標籤
    status_ready: str = "就緒"
    status_running: str = "運行中"
    status_paused: str = "已暫停"
    status_stopped: str = "已停止"
    
    # 工具提示
    tooltip_start: str = "開始模擬"
    tooltip_stop: str = "停止模擬"
    tooltip_pause: str = "暫停模擬"
    tooltip_reset: str = "重置模擬"
'''
    
    # 查找插入位置（在其他@dataclass之后）
    if 'class UILabels:' not in content and 'class UILabels(' not in content:
        # 找到最后一个@dataclass的位置
        last_dataclass_pos = -1
        for match in re.finditer(r'@dataclass\s*\nclass \w+:', content):
            last_dataclass_pos = match.end()
        
        if last_dataclass_pos != -1:
            # 找到这个类的结束位置
            lines = content[last_dataclass_pos:].split('\n')
            class_end = 0
            indent_level = None
            
            for i, line in enumerate(lines):
                if line.strip() == '':
                    continue
                if indent_level is None and line.strip():
                    # 确定缩进级别
                    indent_level = len(line) - len(line.lstrip())
                elif line.strip() and indent_level is not None:
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and not line.startswith(' '):
                        class_end = i
                        break
            
            # 插入UILabels类
            insert_pos = last_dataclass_pos + len('\n'.join(lines[:class_end]))
            content = content[:insert_pos] + '\n' + uilabels_class + '\n' + content[insert_pos:]
        else:
            # 如果找不到合适位置，添加到文件末尾的配置实例之前
            if '# 全域設定實例' in content:
                insert_pos = content.find('# 全域設定實例')
                content = content[:insert_pos] + uilabels_class + '\n\n' + content[insert_pos:]
            else:
                content += '\n' + uilabels_class + '\n'
    
    # 确保有ui_labels实例
    if 'ui_labels = UILabels()' not in content:
        # 查找其他配置实例的位置
        if 'simulator_config = SimulatorConfig()' in content:
            content = content.replace(
                'simulator_config = SimulatorConfig()',
                'simulator_config = SimulatorConfig()\nui_labels = UILabels()'
            )
        elif '# 向後相容的額外設定實例' in content:
            # 在向后兼容设置实例部分添加
            pattern = r'(# 向後相容的額外設定實例.*?)((?:\n\w+_config = \w+\(\))*)'
            replacement = r'\1\2\nui_labels = UILabels()'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            # 添加到文件末尾
            content += '\nui_labels = UILabels()\n'
    
    # 写回文件
    with open(settings_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  UILabels类已添加到settings.py")
    return True

def test_uilabels_import():
    """测试UILabels导入"""
    print("测试UILabels导入...")
    
    try:
        # 清除模块缓存
        import sys
        modules_to_clear = [m for m in sys.modules.keys() if m.startswith('config')]
        for module in modules_to_clear:
            del sys.modules[module]
        
        # 测试导入
        from config.settings import UILabels, ui_labels
        
        print("  [OK] UILabels类导入成功")
        print("  [OK] ui_labels实例导入成功")
        
        # 测试访问属性
        print(f"  测试属性: window_title = '{ui_labels.window_title}'")
        print(f"  测试属性: btn_start = '{ui_labels.btn_start}'")
        
        return True
        
    except ImportError as e:
        print(f"  [ERROR] 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] 测试失败: {e}")
        return False

def show_current_settings_structure():
    """显示当前settings.py的结构"""
    print("显示config/settings.py的类结构...")
    
    settings_file = Path("config/settings.py")
    if not settings_file.exists():
        print("  config/settings.py不存在")
        return
    
    with open(settings_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找所有类定义
    class_pattern = r'^class (\w+).*?:'
    dataclass_pattern = r'^@dataclass\s*\nclass (\w+).*?:'
    
    classes_found = []
    
    # 查找普通类
    for match in re.finditer(class_pattern, content, re.MULTILINE):
        classes_found.append(match.group(1))
    
    # 查找dataclass
    for match in re.finditer(dataclass_pattern, content, re.MULTILINE):
        classes_found.append(match.group(1))
    
    print("  找到的类:")
    for cls_name in sorted(set(classes_found)):
        print(f"    - {cls_name}")
    
    # 检查是否有实例定义
    instance_pattern = r'^(\w+) = (\w+)\(\)'
    instances = re.findall(instance_pattern, content, re.MULTILINE)
    
    if instances:
        print("  找到的配置实例:")
        for var_name, class_name in instances:
            print(f"    - {var_name} = {class_name}()")

def main():
    """主函数"""
    print("验证并修复UILabels问题")
    print("=" * 40)
    
    # 1. 显示当前结构
    show_current_settings_structure()
    
    # 2. 检查UILabels
    has_uilabels = check_uilabels_in_settings()
    
    if not has_uilabels:
        # 3. 添加UILabels
        print("\n修复UILabels...")
        success = add_uilabels_to_settings()
        
        if not success:
            print("[ERROR] 无法添加UILabels到settings.py")
            return
    
    # 4. 测试导入
    print("\n" + "-" * 40)
    import_success = test_uilabels_import()
    
    # 5. 总结
    print("\n" + "=" * 40)
    print("总结:")
    
    if import_success:
        print("[OK] UILabels问题已修复!")
        print("[OK] 现在可以正常导入和使用UILabels")
        print("\n可以运行以下命令验证:")
        print("  python -c \"from config.settings import UILabels; print('UILabels可用')\"")
        print("  python main.py --test")
    else:
        print("[ERROR] UILabels问题仍未解决")
        print("请检查config/settings.py文件内容")

if __name__ == "__main__":
    main()