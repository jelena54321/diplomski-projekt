class Coder:

    GAP = '*'
    ALPHABET = ['A', 'C', 'G', 'T', GAP, 'N']

    encodings = { value: encoding for encoding, value in enumerate(ALPHABET) }
    decodings = { encoding: value for value, encoding in encodings.items() }

    @staticmethod
    def encode(value):
        assert value in Coder.ALPHABET
        return Coder.encodings[value]

    @staticmethod
    def decode(value):
        assert value < len(Coder.ALPHABET)
        return Coder.decodings[value]
