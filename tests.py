import unittest
from USE_TESTING_MCEval import Llama2GenerationPipeline, LabelledDataset, MCEval
import torch

class TestMCEval(unittest.TestCase):
    def test_main(self):
        # Load dataset
        harmless_dataset = LabelledDataset("harmless")
        # Load model
        llama2_pipeline = Llama2GenerationPipeline(
            # device='cpu',
        )
        # Run comparison eval
        mc_eval = MCEval()
        mc_eval.compare_reasoning(harmless_dataset, llama2_pipeline, trial_id=1, debug=True, num_cpus=5)
    # def setUp(self):
    #     self.pipeline = Llama2GenerationPipeline(device='cpu')
    # def test_append_and_tokenize(self):
    #     dataset = LabelledDataset("harmless").dataset
    #     colname = 'input'
    #     new_text = ' Test new text'
    #     # expected_output = {
    #     #     'input_ids': torch.tensor([[101, 7592, 102, 2961, 102]]),
    #     #     'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    #     # }
    #     dataset = dataset.select(range(5))
    #     dataset = dataset.map(lambda x: self.pipeline.append_and_tokenize(x, colname, new_text))

    #     print(dataset)

if __name__ == '__main__':
    unittest.main()