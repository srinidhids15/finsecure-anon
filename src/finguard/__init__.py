"""FinGuard: PII detection and anonymization for financial documents.

Pipeline overview (each arrow passes plain text and/or a list of ``Span``):

    PDF / text  ->  pdf.extract  ->  detection (rules + NER -> hybrid)
                ->  anonymization  ->  safe text  (evaluation scores detection)

See docs/ARCHITECTURE.md for the full description.
"""

from finguard.schema import Span

__version__ = "0.1.0"

__all__ = ["Span", "__version__"]
