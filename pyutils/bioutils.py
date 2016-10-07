__author__ = 'morel'

from itertools import groupby
import re
from shared.pyutils.utils import *

BioUtilComplementDnaDict = {"a":"t", "t":"a", "g":"c", "c":"g", \
    "A":"T", "T":"A", "G":"C", "C":"G"}
BioUtilFastaLineSize = 70

# parsing of a fasta file
# from https://www.biostars.org/p/710/
def BioUtilFastaIter(fastaFileName):
    """
    given a fasta file. yield tuples of header, sequence
    """
    with open(fastaFileName) as fh:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        # we know they alternate.
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">"
            header = header.next()[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.next())
            yield header, seq


# Get nucleotide sequnce out of the full nucleotide file
def BioUtilExtractDnaSeq(fileName, start, stop, oneBased=True, strand='+'):
    firstLine = True
    count = 0
    ret = ""
    if oneBased:
        start -= 1
    else:
        stop += 1

    with open(fileName) as fh:
        for l in fh:
            l = l.strip()
            if (l[0] == '>') and (not firstLine):
                raise ValueError("file %s line %s wrong" % (fileName, l))
            firstLine = False
            oldCount = count
            count += len(l)
            if (start < count) and (stop > oldCount):
                begInd = 0 if (start < oldCount) else (start - oldCount)
                endInd = len(l) if (stop >= count) else (stop - oldCount)
                ret += l[begInd : endInd]

    if strand != '+':
        ret = BioUtilComplementDna(ret)

    return ret


# Returns a complementary strand of dna
def BioUtilComplementDna(dna):
    ret = ""
    for c in dna:
        ret += BioUtilComplementDnaDict[c]
    return ret


# Recording fasta file, from a list of (header, content) tuples
def BioUtilFastaWrite(fileName, itemList):
    with open(fileName, "w") as fh:
        for hdr, s in itemList:
            fh.write(">" + hdr + "\n")
            for i in range(0, len(s), BioUtilFastaLineSize):
                fh.write(s[i : i + BioUtilFastaLineSize] + "\n")


class ProkDnaNameParser(UtilObject):
    """
    Prokaryote DNA Molucule name parser
    chr: chromosome index, 0 based (always 0 for phages and plasmids)
    strain: name of the strain (main if no strain name; None for phages and
        plasmids)
    phage: name of the phage (None if not phage)
    plasmid: name of the plasmid (None if not plasmid)
    isClone - bool, indicating if this is some kind of clone
    """

    patChromosome = re.compile(r'(.*)(\bchromosome\b(?:\s+([^\s,]+))?)(.*)')
    chromosomeXlator = {"i":"1", "ii":"2", "iii":"3"}
    validChromosomes = ["1", "2", "3"]

    # Phage representation (might have no name)
    patPhage = re.compile(r'(.*)(\bphage\b(?:\s+([^\s,]+))?)(.*)')

    # Strain representation (must have name)
    patStrain1 = re.compile(r'(.*)(\bstr\b\.\s+([^\s,]+))(.*)')
    patStrain2 = re.compile(r'(.*)(\bstrain\b\s+([^\s,]+))(.*)')

    # Plasmid and megaplasmid patterns (might have no name)
    patPlasmid1 = \
        re.compile(r'(.*)(\b(?:mega)?plasmid\b(?:\s+([^\s,]+))?)(.*)')
    patPlasmid2 = re.compile(r'(.*)(\bplasmid(\d+))(.*)')

    # Clone match
    patClone = re.compile(r'(.*)(\bclone()\b)(.*)')

    # Element match
    patElement = re.compile(r'(.*)(\b[a-z]+\b\s+\belement()\b)(.*)')


    patWhiteSpace = re.compile(r'\s+')
    patWhiteSpaceComma = re.compile(r'\s+,')
    patQuotedText = re.compile(r'([\'\"](.*?)[\'\"])')
    # "complete chromsome" needs to be removed, otherwise it consumes
    # the next word as the name of this chromosome
    patCompleteChromosome = re.compile(r'complete chromosome ')

    def __init__(self, name):
        self.orgName = name.strip().lower()
        self.name = self.orgName
        self.name = ProkDnaNameParser.patWhiteSpace.sub(' ', self.name)
        self.name = ProkDnaNameParser.patCompleteChromosome.sub('',
            self.name)

        self.chr = None
        self.strain = None
        self.isClone = False
        self.isElement = False
        self.phage = None
        self.plasmid = None

        self.name, temp = self.processPattern(self.name,
            ProkDnaNameParser.patClone, None)
        if temp:
            self.isClone = True

        self.name, self.phage = self.processPattern(self.name,
            ProkDnaNameParser.patPhage, None)
        self.name, self.plasmid = self.processPattern(self.name,
            ProkDnaNameParser.patPlasmid1, None)
        if not self.plasmid:
            self.name, self.plasmid = self.processPattern(self.name,
                ProkDnaNameParser.patPlasmid2, None)
        self.name, temp = self.processPattern(self.name,
            ProkDnaNameParser.patElement, None)
        if temp:
            self.isElement = True

        self.name, self.chr = self.processPattern(self.name,
            ProkDnaNameParser.patChromosome, "1")
        self.xlateChromosomeStr()

        # Some names have an extra chromosome string, remove it
        self.name, _ = self.processPattern(self.name,
            ProkDnaNameParser.patChromosome, None)

        defaultStrainName = "MAIN"
        self.name, self.strain = self.processPattern(self.name,
            ProkDnaNameParser.patStrain1, defaultStrainName)
        if self.strain == defaultStrainName:
            # Try another pattern
            self.name, self.strain = self.processPattern(self.name,
                ProkDnaNameParser.patStrain2, defaultStrainName)

        # Leave only the name up to the comma
        self.name = ProkDnaNameParser.patWhiteSpaceComma.sub(',', self.name)
        self.name = self.name.split(',', 1)[0]

    def xlateChromosomeStr(self):
        # Translates chromosome string if needed
        if self.chr in ProkDnaNameParser.chromosomeXlator:
            self.chr = ProkDnaNameParser.chromosomeXlator[self.chr]

    @staticmethod
    def chromosomeStrToNumber(chr):
        # Returns -1 if this is not a valid chromosome number string
        if chr in ProkDnaNameParser.chromosomeXlator:
            chr = ProkDnaNameParser.chromosomeXlator[chr]
        if chr in ProkDnaNameParser.validChromosomes:
            return int(chr)
        else:
            return -1

    @staticmethod
    def removeQuotes(s):
        miter = ProkDnaNameParser.patQuotedText.finditer(s)
        # Accumulated shift, because replacement string might be of
        # different length
        shift = 0
        for m in miter:
            startPos = m.start(1) + shift
            endPos = m.end(1) + shift
            replStr = m.group(2)
            replStr = ProkDnaNameParser.patWhiteSpace.sub('_', replStr)
            shift += len(replStr) + startPos - endPos
            s = s[:startPos] + replStr + s[endPos:]
        return s

    @staticmethod
    def processPattern(name, pattern, default = None):
        # Extracts matched feature, and returns updated DNA name
        # excluding the pattern, and teh feature value (or default)
        m = pattern.match(name)
        if not m:
            return (name, default)
        featureName = m.group(3)
        if not featureName:
            featureName = "NONAME"
        # Exclude the found feature from the name, and trim white space
        name = ProkDnaNameParser.patWhiteSpace.sub(' ', m.group(1) +
            m.group(4))
        return (ProkDnaNameParser.patWhiteSpaceComma.sub(',', name),
            featureName)





