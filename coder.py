class Coder:
    """
    A class used for encoding and decoding nucleobases.

    The coding alphabet includes four nucleobases (A, C, G and T), a gap (*)
    and 'N' that is used when the information is unknown. 

    """

    GAP = '*'
    UNKNOWN = 'N'
    ALPHABET = ['A', 'C', 'G', 'T', GAP, UNKNOWN]

    encodings = { 'A': 0, 'C': 1, 'G': 2, 'T': 3, GAP: 4, UNKNOWN: 5 }
    decodings = { 0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: GAP, 5: UNKNOWN }

    @staticmethod
    def encode(value):
        """
        Encodes given alphabet element.

        Parameters
        ----------
        value : element of the alphabet

        Returns
        -------
        encoding : encoding of the given parameter

        Raises
        ------
        AssertionError
            If provided value is not the element of the alphabet.
        """

        return Coder.encodings[value]

    @staticmethod
    def decode(value):
        """
        Decodes given alphabet element encoding.

        Parameters
        ----------
        value : encoding of the alphabet element

        Returns
        -------
        decoding : the alphabet element corresponding to the given parameter

        Raises
        ------
        AssertionError
            If provided value is not the encoding of the alphabet element.
        """

        return Coder.decodings[value]
