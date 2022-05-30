from dataclasses import dataclass


@dataclass
class ModelOptions:
    builder_label: str
    model_nn: int
    ingest_dense: int
    output_dense: int
    trainable_from_index: int
    epochs: int
    batch_size: int

    def representation(self):
        parts = [
            f"nn{self.model_nn}",
            f"train{self.trainable_from_index}",
            f"id{self.ingest_dense}",
            f"od{self.output_dense}",
            f"e{self.epochs}",
            f"b{self.batch_size}",
        ]

        return "-".join(parts)