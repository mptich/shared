__author__ = 'morel'

from itertools import groupby

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


