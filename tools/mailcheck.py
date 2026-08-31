#!/usr/bin/env python3
"""Read the trog.co.za catch-all mailbox (the `general` account) read-only.

Used during the manual Ubisoft account-creation batch to fetch signup
verification mails. Uses Stalwart's admin `impersonate` permission over IMAP
SASL PLAIN (authzid=general, authcid=admin) so no per-account password is
needed. Credentials come from configs/mail_admin.local.json (not committed).

Fetches use BODY.PEEK, so messages are never marked read.

    tools/mailcheck.py -n 10                 # 10 most recent, codes + links only
    tools/mailcheck.py --to tmai-03          # only mail addressed to tmai-03@...
    tools/mailcheck.py --to tmai-03 --full   # full body text
"""
import argparse
import email
import imaplib
import json
import os
import re
import sys
from email.header import decode_header

CONF = os.path.join(os.path.dirname(__file__), "..", "configs", "mail_admin.local.json")


def load_conf():
    with open(CONF) as fh:
        return json.load(fh)


def dec(s):
    if not s:
        return ""
    parts = []
    for chunk, enc in decode_header(s):
        if isinstance(chunk, bytes):
            parts.append(chunk.decode(enc or "utf-8", "replace"))
        else:
            parts.append(chunk)
    return "".join(parts)


def connect(c):
    M = imaplib.IMAP4_SSL(c["host"], c.get("imap_port", 993), timeout=20)
    authz = c["impersonate_account"]
    authc = c["admin_user"]
    pw = c["admin_password"]
    M.authenticate("PLAIN", lambda _: f"{authz}\x00{authc}\x00{pw}".encode())
    return M


def body_text(msg):
    cands = []
    if msg.is_multipart():
        for p in msg.walk():
            ct = p.get_content_type()
            if ct in ("text/plain", "text/html"):
                try:
                    cands.append((ct, p.get_payload(decode=True).decode(
                        p.get_content_charset() or "utf-8", "replace")))
                except Exception:
                    pass
    else:
        try:
            cands.append((msg.get_content_type(),
                          msg.get_payload(decode=True).decode(
                              msg.get_content_charset() or "utf-8", "replace")))
        except Exception:
            pass
    for ct, txt in cands:
        if ct == "text/plain":
            return txt
    return cands[0][1] if cands else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=8,
                    help="how many recent messages to scan (default 8)")
    ap.add_argument("-t", "--to", help="only messages whose recipient headers contain this")
    ap.add_argument("--full", action="store_true", help="print full body text")
    ap.add_argument("--folders", action="store_true",
                    help="just list folders + message counts, then exit")
    a = ap.parse_args()

    try:
        c = load_conf()
    except FileNotFoundError:
        sys.exit(f"missing {os.path.abspath(CONF)}")

    M = connect(c)

    if a.folders:
        _, boxes = M.list()
        for b in boxes:
            line = b.decode(errors="replace")
            name = line.split(' "')[-1].strip('"') if '"' in line else line.split()[-1]
            try:
                t, d = M.select(name, readonly=True)
                n = int(d[0]) if t == "OK" else "?"
                M.close()
            except Exception as e:
                n = f"err {e}"
            print(f"{n:>6}  {name}")
        M.logout()
        return

    # Inbound mail to trog.co.za frequently lands in Junk (sender SPF/DKIM/DMARC
    # is often imperfect and Stalwart's spam classifier files it there), so scan
    # Junk as well as INBOX by default.
    folders = ["INBOX", "Junk Mail", "Deleted Items"]
    collected = []  # (epoch_or_seq, folder, raw_bytes)
    for fld in folders:
        t, _ = M.select(fld, readonly=True)
        if t != "OK":
            continue
        _, data = M.search(None, "ALL")
        ids = data[0].split()
        for i in reversed(ids[-40:]):
            _, d = M.fetch(i, "(BODY.PEEK[])")
            if not d or not d[0]:
                continue
            collected.append((int(i), fld, d[0][1]))
        M.close()

    if not collected:
        print("(no messages in INBOX / Junk / Trash)")
        M.logout()
        return

    shown = 0
    for _seq, fld, raw in sorted(collected, key=lambda x: x[0], reverse=True):
        if shown >= a.num:
            break
        msg = email.message_from_bytes(raw)
        recip = " ".join(filter(None, [
            msg.get("To", ""), msg.get("Delivered-To", ""),
            msg.get("X-Original-To", ""), msg.get("Cc", "")]))
        if a.to and a.to.lower() not in recip.lower():
            continue
        shown += 1
        print("=" * 72)
        print("Folder :", fld)
        print("From   :", dec(msg.get("From")))
        print("To     :", dec(msg.get("To")))
        print("Subject:", dec(msg.get("Subject")))
        print("Date   :", msg.get("Date"))
        b = body_text(msg)
        if a.full:
            print("-" * 72)
            print(b.strip()[:6000])
        else:
            codes = list(dict.fromkeys(re.findall(r"(?<!\d)\d{6}(?!\d)", b)))
            links = list(dict.fromkeys(re.findall(r"https?://[^\s\"'<>)\]]+", b)))
            if codes:
                print("CODES  :", ", ".join(codes))
            for l in links:
                if any(k in l.lower() for k in
                       ("ubi", "verify", "valid", "confirm", "activat", "account")):
                    print("LINK   :", l)
    if shown == 0:
        print(f"(no messages matched --to {a.to})")
    M.logout()


if __name__ == "__main__":
    main()
