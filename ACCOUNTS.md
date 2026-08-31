# Fleet accounts — manual signup runbook

8 Ubisoft accounts, one per fleet instance. Credentials live in
`configs/accounts.local.json` (chmod 600, git-ignored). This file is the
*procedure*; that file is the *state*. Flip the booleans there as you go.

## Why this is by hand

One Ubisoft account = one concurrent login, so 8 games need 8 accounts.
Per `FLEET.md`: creating them is the risky step and stays human; a scripted
signup/login loop is indistinguishable from credential stuffing and is the
fastest way to get all 8 banned. Do not automate this. Logging in is also
manual — once per Wine prefix, then the prefix is snapshotted.

## Email

All 8 use `tmai-NN@trog.co.za`. There is no real mailbox per address — the
trog.co.za catch-all rewrites every unknown recipient into the `general`
account. Read it read-only with:

    tools/mailcheck.py --to tmai-00 --full     # full body of mail to that addr
    tools/mailcheck.py -n 10                    # 10 most recent, codes+links only
    tools/mailcheck.py --folders               # sanity check: folder counts

`mailcheck.py` uses the Stalwart admin `impersonate` permission
(`configs/mail_admin.local.json`), so it needs no per-account mail password.

> Known unknown: the `general` mailbox has read empty before despite logged
> deliveries (see the mail-migration project notes). That's why instance 00
> is the canary — confirm its verification mail actually lands before signing
> up the other 7. If it doesn't: check SnappyMail at webmail.trog.co.za, the
> Junk folder, and any Sieve rule.

## Per-account steps

For each row in `accounts.local.json`, in order:

1. **Create the Ubisoft account** — https://account.ubisoft.com → *Create account*.
   - Email = `tmai-NN@trog.co.za`, password = the row's `password`.
   - Date of birth: any adult date. Country: South Africa.
   - Display name: try `display_name_primary`; if taken, walk the
     `display_name_fallbacks`, then invent one and record it in `notes`.
   - Expect an hCaptcha / FunCaptcha. Solve it by hand.
   - Set `ubisoft_account_created: true`.
2. **Verify the email** — Ubisoft sends a 6-digit code or a confirmation link.

       tools/mailcheck.py --to tmai-NN --full

   Enter the code / open the link. Set `email_verified: true`.
3. **Log in inside the instance's Wine prefix** — launch that instance's
   TM2020 once, interactively, and complete the Ubisoft Connect login with
   this account. Set `logged_in_to_prefix: true`.
4. **Set the in-game plate to `TAS`** — Trackmania → Profile → licence plate.
   Hard project constraint: all footage must be self-evidently tool-assisted.
5. **Snapshot the prefix** — the logged-in session lives in the prefix; copy
   it so a container restart never needs a re-login. Set
   `prefix_snapshotted: true`.

## Progress

    python3 -c "import json;a=json.load(open('configs/accounts.local.json'))['accounts'];[print(f\"{x['instance']:>2}  {x['email']:<22} created={x['ubisoft_account_created']!s:<5} verified={x['email_verified']!s:<5} prefix={x['logged_in_to_prefix']!s:<5} snap={x['prefix_snapshotted']}\") for x in a]"
