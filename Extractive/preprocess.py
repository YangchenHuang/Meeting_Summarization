import argparse
import data_builder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-story_path", default='../ext_data/story/', type=str)
    parser.add_argument("-index_path", default='../ext_data/index/', type=str)
    parser.add_argument("-json_path", default='../ext_data/json_data/', type=str)
    parser.add_argument("-bert_path", default='../ext_data/bert_data/', type=str)
    parser.add_argument("-mode", default='sent_dist', type=str, help='distribution or one hot', choices=['sent_dist', 'sent_one'])
    parser.add_argument('-min_src_ntokens_per_sent', default=3, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=300, type=int)



    args = parser.parse_args()

    data_builder.format_to_lines(args)
    data_builder.format_to_bert(args)
