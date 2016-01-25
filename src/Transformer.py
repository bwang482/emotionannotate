# from spellcheck import correct
# Spell checking temporarily removed due to speed issues
import re
import enchant

USERID = "@USERID"
URL = "URL"


class Transformer(object):
    def __init__(self):
        super(Transformer, self).__init__()


class RepeatTransformer(Transformer):
    def __init__(self, aux_dict={}):
        super(RepeatTransformer, self).__init__()
        self.dict = enchant.Dict("en")
        self.aux_dict = aux_dict
        self.repeat_regex = re.compile(r'(\w)\1')

    def transform(self, token):
        while not ((self.dict.check(token)) or (token in self.aux_dict)):
            match = self.repeat_regex.search(token)
            if not match:
                break
            sub = match.groups()[0] if match.groups()[0] is not None else ''
            token = token[:match.start()] + sub + token[match.end():]
        return token


class SubstitutionTransformer(Transformer):
    def __init__(self, substitution_dict):
        super(SubstitutionTransformer, self).__init__()
        self.substitution_dict = substitution_dict
        self.repeat_transformer = RepeatTransformer(self.substitution_dict)

        self.tokeniser = re.compile(r'\A([^\w]*)(.*?)([^\w]*)\Z', re.UNICODE)
        self.word_chars = re.compile(r'[a-zA-Z]', re.UNICODE)
        self.non_word_chars = re.compile(r'[^a-zA-Z]', re.UNICODE)
        self.is_url = re.compile(r'((www\.[\s]+)|(https?://[^\s]+))')
        self.is_id = re.compile(r'@[^\s]+')
        self.is_hashtag = re.compile(r'#[\w]+')

    def transform(self, tweet):
        tokens = self.tokenise(tweet)
        try:
            return ' '.join(map(self.process, tokens))
#        except:
#            print tweet
        except UnicodeDecodeError as e:
            print tokens
            raise e

    def tokenise(self, tweet):
        chunks = tweet.lower().split()
        tokens = []
        for chunk in chunks:
            if self.is_id.match(chunk):
                tokens += [(USERID, False)]
            elif self.is_url.match(chunk):
                tokens += [(URL, False)]
            elif self.is_hashtag.match(chunk):
                tokens += [(chunk, False)]
            elif len(self.word_chars.findall(chunk)) > len(self.non_word_chars.findall(chunk)):
                tokens += [(token, is_word) for token, is_word in zip(self.tokeniser.match(chunk).groups(), [False, True, False]) if token != '']
            else:
                tokens += [(chunk, False)]
        return tokens

    def process(self, (token, is_word)):
        if is_word:
            token = self.repeat_transformer.transform(token)
            if token in self.substitution_dict:
                token = self.substitution_dict[token]
            else:
                # token = correct(token)
                # Spell checking removed due to speed issues
                pass
        return token


class SanitisationTransformer(SubstitutionTransformer):
    def tokenise(self, tweet):
        tokens = []
        for chunk in tweet.split():
            if self.is_id.match(chunk):
                tokens += [USERID]
            elif self.is_url.match(chunk):
                tokens += [URL]
            else:
                tokens += [chunk]
        return tokens

    def process(self, token):
        if token.lower() in self.substitution_dict:
            token = ''
        return token


class HashtagTransformer(Transformer):
    HASH = '#'
    SPACE = ' '

    def transform(self, tweet):
        tokens = tweet.split()
        while len(tokens) > 0 and self.is_hashtag(tokens[-1]):
            tokens.pop()
        if len(tokens) == 0:
            tokens = tweet.split()
        tokens = [word.lstrip(self.HASH) for word in tokens]

        return self.SPACE.join(tokens)

    @staticmethod
    def is_hashtag(token):
        return token[0] == '#'

'''
Used by bullying classifer to efficiently sanitise tweets.
This includes removing stray characters, HTTP links, pic.twitter.com image links.
Also carries out output of bullying data to ARFF format for use in WEKA.
'''


class CharacterSanitiseTransformer(Transformer):

    def __init__(self):
        # Compile Regular expressions for sanitising
        self.regex_replace_pairs = [
            (re.compile(","), " "),
            (re.compile("\t"), ""),
            (re.compile("pic\.twitter\.com[^\s]+\s"), ""),
            (re.compile("\."), ""),
            (re.compile("@[a-zA-Z0-9_]+"), ""),
            (re.compile(" @ "), " at "),
            (re.compile("http[^\s]+\s"), ""),
            (re.compile("[^a-zA-Z0-9#!?*():\s\-=_]"), ""),
            (re.compile("  "), " "),
            (re.compile("\A  "), ""),
            (re.compile("\s"), " ")
        ]

    def transform(self, tweet):
        # Run through regular expressions and find
        sanitised_tweet = tweet + " "
        for regex, replacement in self.regex_replace_pairs:
            sanitised_tweet = regex.sub(replacement, sanitised_tweet)
        return sanitised_tweet.lower()
