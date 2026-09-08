import csv
import json
import os

from boltz_api import Boltz
from dotenv import load_dotenv

load_dotenv()

client = Boltz(api_key=os.environ["BOLTZ_API_KEY"])


with open('sequences/human_b2m.json') as f:
    b2m_sequence = json.load(f)['canonical_sequence']

with open('sequences/hla_a.json') as f:
    hla_a_sequences = json.load(f)

with open('sequences/hla_b.json') as f:
    hla_b_sequences = json.load(f)

with open('sequences/hla_c.json') as f:
    hla_c_sequences = json.load(f)




def run_boltz_prediction(prediction):
    output_dir = f"boltz-experiments/{prediction['file_name']}"
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists, skipping prediction for {prediction['file_name']}.")
        return

    run_dir = client.experiments.run_structure_and_binding(
        entities=[
            {"type": "protein", "value": prediction['allele_sequence'], "chain_ids": ["A"]},
            {"type": "protein", "value": prediction['b2m_sequence'], "chain_ids": ["B"]},
            {"type": "protein", "value": prediction['peptide_sequence'], "chain_ids": ["C"]},
        ],
        model="boltz-2.1",
        name=prediction['file_name'],
    )
    print(f"Prediction for {prediction['file_name']} submitted, run directory: {run_dir}")


complexes = []
predictions = []

with open('complexes/hla_class_i.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pdb_id, locus, allele_slug, peptide_sequence, resolution = row
        file_name = f"{pdb_id}__{allele_slug}__{peptide_sequence}"
        allele_sequence = None
        if locus == 'hla-a':
            allele_sequence = hla_a_sequences[allele_slug]['canonical_sequence']
        elif locus == 'hla-b':
            allele_sequence = hla_b_sequences[allele_slug]['canonical_sequence']
        elif locus == 'hla-c':
            allele_sequence = hla_c_sequences[allele_slug]['canonical_sequence']

        if allele_sequence is not None:
            complex = f"{allele_sequence}_{peptide_sequence}"
            if complex not in complexes:
                complexes.append(complex)
                print(f"Complex {allele_slug} with peptide {peptide_sequence} added, processing.")
                prediction = {
                    "file_name": file_name,
                    "peptide_sequence": peptide_sequence,
                    "allele_sequence": allele_sequence,
                    "b2m_sequence": b2m_sequence,
                }
                predictions.append(prediction)
            else:
                print(f"Complex {allele_slug} with peptide {peptide_sequence} already processed, skipping.")
                continue
        

for prediction in predictions:
    run_boltz_prediction(prediction)
