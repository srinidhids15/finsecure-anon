# data/

| Folder | Contents | In Git? |
|---|---|---|
| `raw/` | Datasets exactly as downloaded | No (ignored) |
| `processed/` | Cleaned and split data produced by FinGuard code | No (ignored) |
| `private/` | Real or sensitive documents for local testing | **Never** |
| `samples/` | Small, hand-written, fully synthetic examples | Yes |

Processed data must be reproducible from `raw/` by running project code.
Never put a real person's document in `samples/`.
