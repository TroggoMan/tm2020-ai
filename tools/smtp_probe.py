#!/usr/bin/env python3
"""Probe inbound SMTP delivery to trog.co.za. Connects to the MX on port 25,
runs a full SMTP transaction to a given recipient, and (unless --no-send)
delivers a short test message. Used to diagnose why external mail to
tmai-NN@trog.co.za / catch-all addresses isn't arriving."""
import argparse
import smtplib
import sys

MX = "mail.trog.co.za"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rcpt", help="recipient, e.g. bingle@trog.co.za")
    ap.add_argument("--from", dest="sender", default="probe@example.com")
    ap.add_argument("--no-send", action="store_true",
                    help="stop after RCPT TO (no DATA)")
    a = ap.parse_args()

    s = smtplib.SMTP(MX, 25, timeout=20)
    s.set_debuglevel(1)
    code, msg = s.ehlo("probe.example.com")
    print("EHLO:", code)
    try:
        if s.has_extn("starttls"):
            s.starttls()
            s.ehlo("probe.example.com")
    except Exception as e:
        print("STARTTLS skipped:", e)

    code, msg = s.mail(a.sender)
    print("MAIL FROM:", code, msg.decode(errors="replace"))
    code, msg = s.rcpt(a.rcpt)
    print("RCPT TO:", code, msg.decode(errors="replace"))
    if code >= 400:
        print(">>> recipient REJECTED at RCPT stage")
        s.quit()
        sys.exit(1)

    if a.no_send:
        print(">>> recipient ACCEPTED at RCPT (no message sent)")
        s.quit()
        return

    body = (f"From: {a.sender}\r\nTo: {a.rcpt}\r\n"
            f"Subject: smtp_probe test\r\n\r\n"
            f"Probe delivery test to {a.rcpt}.\r\n")
    code, msg = s.data(body)
    print("DATA:", code, msg.decode(errors="replace"))
    s.quit()
    print(">>> message accepted for delivery" if code < 400 else ">>> DATA rejected")


if __name__ == "__main__":
    main()
