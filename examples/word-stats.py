import json
import sys
import logging

from polymr import Polymr

def main(fname):
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    # Share a root
    words = Polymr.text(fname) \
            .flat_map(lambda line: line.split())

    # Total number of words seen
    total_count = words.count(lambda word: None)

    # Most frequent words
    top_words = words.count(lambda word: word) \
              .sort_by(lambda word_count: -word_count[1])

    # Character lengths
    word_lengths = words.count(lambda word: len(word)) \
            .sort_by(lambda cl: cl[0])

    # Average character length
    avg_word_lengths = word_lengths \
            .map(lambda wl: wl[0] * wl[1]) \
            .a_group_by(lambda x: None) \
                .sum() \
            .join(total_count) \
                .reduce(lambda awl, tc: next(awl)[1] / float(next(tc)[1]))

    tc, tw, wl, awl = Polymr.run(total_count, top_words, word_lengths, 
            avg_word_lengths, name="word-stats")

    for _, v in tc:
        print("Total Count: ", v)

    print("Top 10 words")
    for word, count in tw.read(10):
        print(word, count)

    print("Character histogram")
    for cl, length in wl.read(20):
        print(cl, length)

    print("Average Word Length: ", awl.read(1)[0][1])

if __name__ == '__main__':
    main(sys.argv[1])
