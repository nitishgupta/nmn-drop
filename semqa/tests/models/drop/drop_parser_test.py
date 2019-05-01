#pylint: disable=unused-import
from flaky import flaky

from allennlp.common.testing import ModelTestCase

from semqa.data.dataset_readers.drop import drop_reader_old
from semqa.models.drop import drop_parser_old
from semqa.domain_languages.drop_old.drop_language import DropLanguage, Date

class DROPSemanticParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        print(self.FIXTURES_ROOT)
        self.set_up_model("semqa/tests/fixtures/drop_parser/experiment.json",
                          "semqa/tests/data/drop_old/date/drop_old.json")

    '''
    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
    '''

    @flaky
    def test_drop_dates(self):

        dates = [
                    Date(year=1354, month=-1, day=-1),
                    Date(year=1364, month=-1, day=-1),
                    Date(year=1364, month=5, day=7),
                ]

        date_mat = DropLanguage.compute_date_comparison_matrices(date_values=dates, device_id=-1)

        print(date_mat)

        assert 1 == 1

