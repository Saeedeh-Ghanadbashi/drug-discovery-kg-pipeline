from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen import predict  # Correct import
import numpy as np
import pandas as pd

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
        num_epochs=300,  # More epochs for better convergence
        batch_size=32
    ),
    model_kwargs=dict(
        embedding_dim=50  # Dimension of embeddings
    ),
    random_seed=42  # For reproducibility
)

# Print evaluation results
print("\nEvaluation Results:")
print(f"Mean Rank: {result.metric_results.get_metric('mean_rank')}")
print(f"Hits@10: {result.metric_results.get_metric('hits_at_10')}")

# Get predictions for BRCA1 gene associations using the correct function
print("\nPredicted disease associations for BRCA1:")
try:
    # Method 1: Using the predict module (for newer versions)
    pred_df = predict.get_tail_prediction_df(
        result.model,
        head_label="BRCA1",
        relation_label="associated_with",
        triples_factory=result.training,
        add_novelties=True
    )
except AttributeError:
    # Method 2: Alternative approach if the above doesn't work
    from pykeen.predict import predict_target
    predictions = predict_target(
        model=result.model,
        head="BRCA1",
        relation="associated_with",
        triples_factory=result.training
    )
    pred_df = predictions.df()
    
print(pred_df.head(10))

# Alternative prediction method that should work across versions
def get_predictions(model, head, relation, triples_factory, k=10):
    """Alternative method to get predictions"""
    # Get all entity IDs
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
    
    # Convert to dataframe
    scores_np = scores.numpy()
    entities = list(entity_to_id.keys())
    
    result_df = pd.DataFrame({
        'tail_label': entities,
        'score': scores_np
    })
    
    # Sort by score (descending)
    result_df = result_df.sort_values('score', ascending=False)
    
    return result_df.head(k)

# Get predictions using alternative method
print("\nAlternative method - Predicted disease associations for BRCA1:")
try:
    import torch
    pred_df_alt = get_predictions(
        result.model, 
        "BRCA1", 
        "associated_with", 
        result.training
    )
    print(pred_df_alt.head(10))
except ImportError:
    print("PyTorch not available for alternative method")

# Save the model for future use
result.save_to_directory('./drug_discovery_model')

print("\nModel saved to './drug_discovery_model/'")