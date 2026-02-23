import torch
import numpy as np
import collections
from cnn import SimpleRamanCNN

# 1. DICTIONARIES
tax_treat_mapping = {
    # Group: Meropenem
    3: {"name": "E. coli 1", "treatment": "Meropenem"},
    4: {"name": "E. coli 2", "treatment": "Meropenem"},
    9: {"name": "K. pneumoniae 1", "treatment": "Meropenem"},
    10: {"name": "K. pneumoniae 2", "treatment": "Meropenem"},
    2: {"name": "K. aerogenes", "treatment": "Meropenem"},
    8: {"name": "E. cloeacae", "treatment": "Meropenem"},
    11: {"name": "P. mirabilis", "treatment": "Meropenem"},
    22: {"name": "S. marcescens", "treatment": "Meropenem"},
    # Group: Piperacillin-tazobactam (TZP)
    12: {"name": "P.aeruginosa 1", "treatment": "TZP"},
    13: {"name": "P. aeruginosa 2", "treatment": "TZP"},
    # Group: Vancomycin
    14: {"name": "MSSA 1", "treatment": "Vancomycin"},
    18: {"name": "MSSA 2", "treatment": "Vancomycin"},
    15: {"name": "MSSA 3", "treatment": "Vancomycin"},
    20: {"name": "S. epidermidis", "treatment": "Vancomycin"},
    21: {"name": "S. lugdunensis", "treatment": "Vancomycin"},
    16: {"name": "MRSA 1 (isogenic)", "treatment": "Vancomycin"},
    17: {"name": "MRSA 2", "treatment": "Vancomycin"},
    # Group: Ceftriaxone
    23: {"name": "S. pneumoniae 2", "treatment": "Ceftriaxone"},
    24: {"name": "S. pneumoniae 1", "treatment": "Ceftriaxone"},
    # Group: Penicillin
    26: {"name": "Group A Strep.", "treatment": "Penicillin"},
    27: {"name": "Group B Strep.", "treatment": "Penicillin"},
    28: {"name": "Group C Strep.", "treatment": "Penicillin"},
    29: {"name": "Group G Strep.", "treatment": "Penicillin"},
    25: {"name": "S. sanguinis", "treatment": "Penicillin"},
    6: {"name": "E. faecalis 1", "treatment": "Penicillin"},
    7: {"name": "E. faecalis 2", "treatment": "Penicillin"},
    # Group: Daptomycin
    5: {"name": "E. faecium", "treatment": "Daptomycin"},
    # Group: Ciprofloxacin
    19: {"name": "S. enterica", "treatment": "Ciprofloxacin"},
    # Group: Caspofungin
    0: {"name": "C. albicans", "treatment": "Caspofungin"},
    1: {"name": "C. glabrata", "treatment": "Caspofungin"}
}


clinical_to_model_mapping = {
    6: 5, 5: 7, 3: 17, 2: 26, 0: 3 
}

# 2. FUNCTION TO EVALUATE MODEL FOR DIFFERENT DATA
def evaluate_model(model, X_data, y_data, device, label_mapping=None, title="TEST"):
    # Transforming into tensors an Z-Normalizing
    spectra = torch.from_numpy(X_data).float()
    spectra = torch.stack([(e - e.mean()) / e.std() for e in spectra])
    spectra = spectra.unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(spectra)
        preds = torch.argmax(out, 1).cpu().numpy()

    species_correct = 0
    treatment_correct = 0
    total = len(y_data)

    for i in range(total):
        real_id = int(y_data[i])
        pred_id = int(preds[i])
        # Apply the label translator (mapping) only if it exists (in the clinical case)
        expected_id = label_mapping.get(real_id, real_id) if label_mapping else real_id
        
        if pred_id == expected_id:
            species_correct += 1
            
        if tax_treat_mapping[expected_id]['treatment'] == tax_treat_mapping[pred_id]['treatment']:
            treatment_correct += 1

    print(f"\n--- {title} ---")
    print(f"Taxonomic Accuracy: {100 * species_correct / total:.2f}%")
    print(f"Clinical Accuracy (ATB): {100 * treatment_correct / total:.2f}%")
    return preds

# 3. EXECUTION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRamanCNN().to(device)
model.load_state_dict(torch.load("simple_raman_cnn.pth", map_location=device))
    
# --- TEST 1: Laboratory test split ---
print("Evaluating Model for Laboratory Data ...")
X_test = np.load('data/X_test_split.npy')
y_test = np.load('data/y_test_split.npy')
evaluate_model(model, X_test, y_test, device, title="LAB TEST")

# --- TEST 2: Clinical test ---
print("\nEvaluatind Model for Clinical Data...")
X_2019 = np.load('data/X_2019clinical.npy')
y_2019 = np.load('data/y_2019clinical.npy')
clinical_predictions = evaluate_model(model, X_2019, y_2019, device, 
                                   label_mapping=clinical_to_model_mapping, 
                                   title="CLINICAL TEST")
