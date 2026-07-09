from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_transformation import DataTransformation


class DataTransformationPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name).get_data_transformation_config()

    def run(self):
        DataTransformation(self.cfg).initiate()


if __name__ == "__main__":
    root_cfg = ConfigurationManager()
    all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
    print(f"Datasets to transform: {all_datasets}")
    for ds in all_datasets:
        print(f"\n--- {ds} ---")
        DataTransformationPipeline(dataset_name=ds).run()
