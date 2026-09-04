"""Recommend a pipeline for kc1-binary (OpenML id 1066)."""
from beta import BETA

model = BETA.from_pretrained()
result = model.recommend(openml_id=1066)

print("pipeline:", result.pipeline)
print("selected via:", result.source, "score:", result.score)
print("\ntransferred heuristic (eta) per step:")
for step, values in result.eta.items():
    print(f"  {step}: {values}")

print("\ntop retrieved neighbors:")
for dataset_id, similarity in result.neighbors:
    print(f"  {dataset_id}: {similarity:.4f}")
