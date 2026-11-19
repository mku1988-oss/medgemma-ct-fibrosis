"""
Training dataset with dynamic slice sampling.

Each item corresponds to one patient, not one slice. Every time a patient is
drawn, a different lower-lobe slice is picked at random, so over several
epochs the model sees many slices per patient while each training example
still holds a single image (which keeps memory use low).

Fibrosis patients can be oversampled to balance the classes.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from torch.utils.data import Dataset

from .prompt import answer_for_label, build_messages

logger = logging.getLogger(__name__)


class SliceDataset(Dataset):
    """
    Args:
        manifest_path: manifest.json written by preprocessing/build_dataset.py.
        patient_ids: Patients to include (for example, the training split).
            None includes everyone in the manifest.
        fibrosis_oversampling: How many times each fibrosis patient appears
            per epoch. 1 means no oversampling.
    """

    def __init__(self,
                 manifest_path: Path,
                 patient_ids: Optional[Iterable[str]] = None,
                 fibrosis_oversampling: int = 1):
        manifest_path = Path(manifest_path)
        self.data_root = manifest_path.parent
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))

        wanted = set(map(str, patient_ids)) if patient_ids is not None else None
        self.slices_by_patient: Dict[str, List[str]] = defaultdict(list)
        self.label_by_patient: Dict[str, int] = {}
        for entry in entries:
            pid = str(entry["patient_id"])
            if wanted is None or pid in wanted:
                self.slices_by_patient[pid].append(entry["image"])
                self.label_by_patient[pid] = int(entry["label"])

        fibrosis = [p for p, label in self.label_by_patient.items() if label == 1]
        normal = [p for p, label in self.label_by_patient.items() if label == 0]
        self.sampling_list = sorted(normal) + sorted(fibrosis) * max(1, fibrosis_oversampling)

        logger.info("SliceDataset: %d patients (%d fibrosis, %d normal), %d items per epoch",
                    len(self.label_by_patient), len(fibrosis), len(normal), len(self.sampling_list))

    def __len__(self) -> int:
        return len(self.sampling_list)

    def __getitem__(self, index: int) -> Dict:
        patient_id = self.sampling_list[index]
        image_path = str(self.data_root / random.choice(self.slices_by_patient[patient_id]))
        answer = answer_for_label(self.label_by_patient[patient_id])

        messages = build_messages(image_path)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return {"messages": messages}
