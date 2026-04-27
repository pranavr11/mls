import json
p = "baselines.ipynb"
with open(p) as f:
    nb = json.load(f)
with open("_nb_cells_out.txt", "w") as out:
    out.write(f"cells {len(nb['cells'])}\n")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "sw_model = AutoModelForCausalLM" in src:
            out.write(f"sw_load_idx={i}\n")
        if "# # 9d) Training loop (sliding-window model)" in src:
            out.write(f"sw_train_idx={i}\n")
