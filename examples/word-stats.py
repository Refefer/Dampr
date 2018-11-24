import json
import sys
import logging

from polymr import Polymr

def main(fname):
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s')

    # Share a root
    words = Polymr.text(fname, 1024**2) \
            .flat_map(lambda line: line.split())

    # Most frequent words
    top_words = words.count(lambda x: x) \
              .sort_by(lambda word_count: -word_count[1])

    # Total number of words seen
    total_count = top_words.fold_by(
            key=lambda word: 1, 
            value=lambda x: x[1],
            binop=lambda x,y: x + y)

    # Character lengths
    word_lengths = top_words \
            .fold_by(lambda tc: len(tc[0]), 
                    value=lambda tc: tc[1], 
                    binop=lambda x,y: x+y) \
            .sort_by(lambda cl: cl[0])

    # Average character length
    avg_word_lengths = word_lengths \
            .map(lambda wl: wl[0] * wl[1]) \
            .a_group_by(lambda x: 1) \
                .sum() \
            .join(total_count) \
                .reduce(lambda awl, tc: next(awl)[1] / float(next(tc)[1]))

    tc, tw, wl, awl = Polymr.run(total_count, top_words, word_lengths, 
            avg_word_lengths, name="word-stats")

    print()
    print("*" * 10)
    print("Word Stats")
    print("*" * 10)
    print("Total Words Found: ", tc.read(1)[0][1])

    print("\nTop 10 words")
    print("\n************")
    for word, count in tw.read(10):
        print(word, count)

    print("\nCharacter histogram")
    print("\n*******************")
    for cl, length in wl.read(20):
        print(cl, length)

    print("\nAverage Word Length: ", awl.read(1)[0][1])

if __name__ == '__main__':
    main(sys.argv[1])
