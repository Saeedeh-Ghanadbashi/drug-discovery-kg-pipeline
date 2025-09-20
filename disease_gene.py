from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
import pandas as pd
import torch

# Create a more comprehensive dataset for drug discovery
triples = [
    # Gene-Disease associations
    ("BRCA1", "associated_with", "Breast_Cancer"),
    ("TP53", "associated_with", "Lung_Cancer"),
    ("EGFR", "associated_with", "Lung_Cancer"),
    ("BRCA1", "associated_with", "Ovarian_Cancer"),
    ("APOE", "associated_with", "Alzheimer"),
    ("HER2", "associated_with", "Breast_Cancer"),
    ("KRAS", "associated_with", "Colorectal_Cancer"),
    ("BRAF", "associated_with", "Melanoma"),
    
    # Drug-Gene interactions
    ("Trastuzumab", "targets", "HER2"),
    ("Gefitinib", "targets", "EGFR"),
    ("Olaparib", "targets", "BRCA1"),
    ("Vemurafenib", "targets", "BRAF"),
    
    # Drug-Disease treatments
    ("Trastuzumab", "treats", "Breast_Cancer"),
    ("Gefitinib", "treats", "Lung_Cancer"),
    ("Olaparib", "treats", "Ovarian_Cancer"),
    ("Vemurafenib", "treats", "Melanoma"),
    
    # Additional relationships for better model training
    ("BRCA2", "associated_with", "Breast_Cancer"),
    ("BRCA2", "associated_with", "Ovarian_Cancer"),
    ("PSEN1", "associated_with", "Alzheimer"),
    ("Tau", "associated_with", "Alzheimer"),
]

# Convert to numpy array and create TriplesFactory
triples_array = np.array(triples)
tf = TriplesFactory.from_labeled_triples(triples_array)

print(f"Total triples: {len(triples)}")
print(f"Number of entities: {tf.num_entities}")
print(f"Number of relations: {tf.num_relations}")

# Split data properly (80% training, 10% testing, 10% validation)
training, testing, validation = tf.split([0.8, 0.1, 0.1])

# Run the pipeline with appropriate parameters
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model="TransE",
    training_kwargs=dict(
        num_epochs=100,
        batch_size=32
    ),
    model_kwargs=dict(
        embedding_dim=50
    ),
    random_seed=42
)

# Print evaluation results
print("\nEvaluation Results:")
print(f"Mean Rank: {result.metric_results.get_metric('mean_rank')}")
print(f"Hits@10: {result.metric_results.get_metric('hits_at_10')}")

# Fixed prediction function
def get_predictions(model, head, relation, triples_factory, k=10):
    """Get top-k predictions for (head, relation, ?)"""
    # Get mapping dictionaries
    entity_to_id = triples_factory.entity_to_id
    relation_to_id = triples_factory.relation_to_id
    
    # Convert labels to IDs
    head_id = entity_to_id[head]
    relation_id = relation_to_id[relation]
    
    # Create batch for prediction
    batch = torch.zeros(len(entity_to_id), 3, dtype=torch.long)
    batch[:, 0] = head_id
    batch[:, 1] = relation_id
    batch[:, 2] = torch.arange(len(entity_to_id))
    
    # Get scores
    with torch.no_grad():
        scores = model.score_hrt(batch)
    
    # Convert to 1D numpy array
    scores_np = scores.cpu().numpy().flatten()  # Ensure it's 1-dimensional
    entities = list(entity_to_id.keys())
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'tail_label': entities,
        'score': scores_np
    })
    
    # Sort by score (descending) and return top k
    return result_df.sort_values('score', ascending=False).head(k)

# Get predictions for BRCA1 gene associations
print("\nPredicted disease associations for BRCA1:")
try:
    pred_df = get_predictions(
        result.model, 
        "BRCA1", 
        "associated_with", 
        result.training
    )
    print(pred_df)
except Exception as e:
    print(f"Error getting predictions: {e}")

# Try alternative prediction approach
try:
    # Simple approach: use the model's prediction methods directly
    from pykeen.models import predict_target
    
    predictions = predict_target(
        model=result.model,
        head="BRCA1",
        relation="associated_with",
        triples_factory=result.training
    )
    
    print("\nUsing predict_target:")
    print(predictions.df.head(10))
    
except Exception as e:
    print(f"\nError with predict_target: {e}")

# Save the model for future use
result.save_to_directory('./drug_discovery_model')
print("\nModel saved to './drug_discovery_model/'")

# Additional: Show some example predictions
print("\nExample predictions:")
diseases_to_predict = ["BRCA1", "TP53", "EGFR"]
for gene in diseases_to_predict:
    try:
        predictions = get_predictions(
            result.model, 
            gene, 
            "associated_with", 
            result.training,
            k=5
        )
        print(f"\nTop diseases associated with {gene}:")
        for _, row in predictions.iterrows():
            print(f"  {row['tail_label']}: {row['score']:.4f}")
    except Exception as e:
        print(f"Error predicting for {gene}: {e}")
        continue

# Show model performance metrics
print(f"\nModel Performance:")
print(f"Training loss: {result.losses[-1] if result.losses else 'N/A'}")
print(f"Number of training epochs: {len(result.losses) if result.losses else 'N/A'}")

# Show what diseases are already known to be associated with BRCA1
print(f"\nKnown associations for BRCA1:")
known_associations = [(h, r, t) for h, r, t in triples if h == "BRCA1" and r == "associated_with"]
for h, r, t in known_associations:
    print(f"  {h} {r} {t}")
