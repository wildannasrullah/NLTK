import nltk
#nltk.download('toolbox')
from nltk.toolbox import ToolboxData
from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *


class ToolboxCorpusReader(CorpusReader):
    def xml(self, fileids, key=None):
        return concat(
            [
                ToolboxData(path, enc).parse(key=key)
                for (path, enc) in self.abspaths(fileids, True)
            ]
        )


    def fields(
        self,
        fileids,
        strip=True,
        unwrap=True,
        encoding="utf8",
        errors="strict",
        unicode_fields=None,
    ):
        return concat(
            [
                list(
                    ToolboxData(fileid, enc).fields(
                        strip, unwrap, encoding, errors, unicode_fields
                    )
                )
                for (fileid, enc) in self.abspaths(fileids, include_encoding=True)
            ]
        )


    # should probably be done lazily:
    def entries(self, fileids, **kwargs):
        if "key" in kwargs:
            key = kwargs["key"]
            del kwargs["key"]
        else:
            key = "lx"  # the default key in MDF
        entries = []
        for marker, contents in self.fields(fileids, **kwargs):
            if marker == key:
                entries.append((contents, []))
            else:
                try:
                    entries[-1][-1].append((marker, contents))
                except IndexError:
                    pass
        return entries


    def words(self, fileids, key="lx"):
        return [contents for marker, contents in self.fields(fileids) if marker == key]


    def raw(self, fileids):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])



def demo():
    pass



if __name__ == "__main__":
    demo()
