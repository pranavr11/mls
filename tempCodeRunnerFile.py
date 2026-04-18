from datasets import load_dataset
raise Exception("running")
cache_dir = "./data"
pg19 = load_dataset("pg19", cache_dir=cache_dir)
scrolls = load_dataset("tau/scrolls", cache_dir=cache_dir)

print(pg19)
print(scrolls)