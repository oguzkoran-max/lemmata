# Sample Texts: Italo Svevo's Trieste Trilogy

Three novels by Italo Svevo (1861-1928), collectively known as the Trieste Trilogy. Clean plain-text files with metadata stripped; only novel text remains.

## Files

| File | Title | Words | Description |
|------|-------|------:|-------------|
| `svevo_la_coscienza_di_zeno.txt` | *La coscienza di Zeno* (1923) | 141,902 | Svevo's masterpiece; unreliable narrator Zeno Cosini recounts his life to a psychoanalyst |
| `svevo_senilita.txt` | *Senilita* (1898) | 65,681 | Emilio Brentani's self-deceptive affair in Trieste |
| `svevo_una_vita.txt` | *Una vita* (1892) | 114,252 | Alfonso Nitti's failed ambitions in a Trieste banking house |

**Total corpus:** 321,835 words

## Suggested Lemmata Parameters

- **Language:** Italian (it)
- **Number of Topics:** 10 (Lemmata's default)
- **Chunk Size:** 1000 (Lemmata's default)
- **Words per Topic:** 10

With "Split every file into parts" ticked (the default), every file longer
than the chunk size is split into parts of about that many words. Uploaded
together, the three novels give 327 documents (La coscienza di Zeno 144,
Una vita 116, Senilità 67), and the default 10 topics can be fitted. Untick
the box to treat each novel as one document; with three documents Lemmata
allows at most 2 topics.

## Source

Digital editions from [LiberLiber](https://www.liberliber.it). Public domain works; digital editions released under the LiberLiber license.
