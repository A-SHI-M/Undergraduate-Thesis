from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_ingestion import DataIngestion


class DataIngestionPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name).get_data_ingestion_config()

    def run(self):
        DataIngestion(self.cfg).download()


if __name__ == "__main__":
    root_cfg = ConfigurationManager()
    all_datasets = list(root_cfg.config["data_ingestion"]["datasets"].keys())
    print(f"Datasets to download: {all_datasets}")
    for ds in all_datasets:
        print(f"\n--- {ds} ---")
        DataIngestionPipeline(dataset_name=ds).run()
