from loader.classe_loader import DataLoader
from loader.csv_loader import CsvLoader
from loader.xml_loader import XmlLoader
from loader.json_loader import JsonLoader


class Factory:
    @staticmethod
    def get_loader(file_path: str) -> DataLoader:
        if file_path.endswith('.csv'):
            return CsvLoader()
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return XmlLoader()
        elif file_path.endswith('.json'):
            return JsonLoader()
        else:
            raise ValueError(f"Formato file non supportato: {file_path}")