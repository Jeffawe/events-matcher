"""Language- and bank-specific wording, kept in ONE place so another language or bank is a
change to this file, not to the matching code.

These are the only places the pipeline reads English words from memos:
  BANK_ALERT     a memo that is really the bank's own notification of a payment
  SAYS_SEPARATE  the sender states this is a separate payment, not a repeat
  NOT_INCOMING   the line is money going out or a request, not money received
Phonetic matching in level2_variants.phonetic() is likewise tuned to English spelling; set
level2_variants.USE_PHONETIC = False for names in other languages.
"""

import re

BANK_ALERT = re.compile(r"\balert\b|you received|deposit received", re.I)
SAYS_SEPARATE = re.compile(r"separate|not a duplicate|another payment", re.I)
NOT_INCOMING = re.compile(r"\byou paid\b|\brequests?\b", re.I)
