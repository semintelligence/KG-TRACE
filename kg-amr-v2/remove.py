import os
import re

def remove():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Content Replacement
    # Replacements (case-sensitive to be safe)
    replacements = [
        (r'KG-AMR', 'KG-AMR'),
        (r'KG-AMR', 'KG-AMR'),
        (r'KG-AMR', 'KG-AMR'),
        (r'KGAMR', 'KGAMR'),
        (r'KG-AMR', 'kg-amr'),
        (r'kg_amr', 'kg_amr'),
        (r'KG-AMR', 'KG-AMR'),
        (r'best', 'best'),
        (r'KGAMR', 'KGAMR'),
        (r'', ''),
        (r'', ''),
    ]

    for root, dirs, files in os.walk(base_dir):
        if '.git' in root or '__pycache__' in root or 'venv' in root or 'miniconda' in root:
            continue
            
        for file in files:
            if file.endswith(('.py', '.json', '.md', '.html', '.tex', '.yml', '.txt', '.csv', '.sh')):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    orig_content = content
                    for pattern, replacement in replacements:
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                        
                    if orig_content != content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"Updated content in {filepath}")
                except Exception as e:
                    pass

    # 2. File/Folder Renaming
    for root, dirs, files in os.walk(base_dir, topdown=False):
        if '.git' in root or '__pycache__' in root or 'venv' in root or 'miniconda' in root:
            continue
            
        for name in files + dirs:
            if 'v2' in name.lower() or 'v_2' in name.lower():
                old_path = os.path.join(root, name)
                new_name = re.sub(r'[_|-]?v2', '', name, flags=re.IGNORECASE)
                new_path = os.path.join(root, new_name)
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} -> {new_path}")

if __name__ == "__main__":
    remove()
