import argparse
import data_builder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tokenizer", default='bert', type=str)
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-story_path", default='../ext_data/result_story/', type=str)
    parser.add_argument("-json_path", default='../abs_data/json_data/', type=str)
    parser.add_argument("-bert_path", default='../abs_data/bert_data/', type=str)
    parser.add_argument("-long_path", default='../abs_data/long_data/', type=str)
    parser.add_argument('-min_src_nsents', default=3, type=int)
    parser.add_argument('-max_src_nsents', default=300, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=200, type=int)


    args = parser.parse_args()

    if args.tokenizer == 'longformer':
        data_builder.format_to_lines(args)
        data_builder.format_to_longformer(args)
    elif args.tokenizer == 'bert':
        data_builder.format_to_lines(args)
        data_builder.format_to_bert(args)
