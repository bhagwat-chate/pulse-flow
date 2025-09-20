import os

# Get all tracked files
files = os.popen("git ls-files --directory").read().splitlines()

# Filter out __init__.py
files = [f for f in files if not f.endswith("__init__.py")]

# Build tree
tree = {}
for f in files:
    parts = f.split("/")
    cur = tree
    for p in parts:
        cur = cur.setdefault(p, {})

def print_tree(d, indent=""):
    for i, (k, v) in enumerate(sorted(d.items())):
        connector = "└── " if i == len(d) - 1 else "├── "
        print(indent + connector + k)
        if v:
            print_tree(v, indent + ("    " if i == len(d) - 1 else "│   "))

print_tree(tree)
