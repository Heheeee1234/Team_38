import json, tempfile, unittest
from pathlib import Path
from clinical_data_gen.cli import validate_directory
from clinical_data_gen.generator import ClinicalDataGenerator, write_jsonl
class GeneratorTests(unittest.TestCase):
 def test_seed_recreates_events(self):
  c=json.loads(Path("config/generator_config.json").read_text()); self.assertEqual(ClinicalDataGenerator(7,c).create(4,30)["vitals_observed"],ClinicalDataGenerator(7,c).create(4,30)["vitals_observed"])
 def test_live_events_do_not_expose_labels(self):
  c=json.loads(Path("config/generator_config.json").read_text()); data=ClinicalDataGenerator(8,c).create(5,20)
  with tempfile.TemporaryDirectory() as d:
   for name,value in data.items(): write_jsonl(Path(d)/f"{name}.jsonl",value)
   self.assertEqual(validate_directory(Path(d))["status"],"pass")
if __name__=="__main__": unittest.main()
