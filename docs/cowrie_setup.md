# Cowrie honeypot setup

Step-by-step setup for the HoneyAI data-collection honeypot. Follow it on a cheap,
disposable VPS. The honeypot catches real SSH attackers; its logs become the
project's **unlabelled wild traffic** (used for distribution comparison, case
studies and the demo, never for reported accuracy).

## 0. Provision a server

- Any VPS provider (DigitalOcean, Vultr, Linode), ~US$5/month, destroyable later.
- **Ubuntu 22.04 LTS**, 1 GB RAM is plenty, region near Singapore.
- Note the public IP.

If your network (e.g. campus) blocks outbound port 22, use the provider's **web
console** for all admin, or move admin SSH to a non-22 port (below) which such
networks usually allow.

## 1. Free port 22 for the honeypot

Move your real admin SSH off port 22 so Cowrie can use it:

```bash
sed -i 's/^#Port 22/Port 2277/' /etc/ssh/sshd_config
systemctl restart ssh
```

From now on you log in with `ssh -p 2277 root@<IP>`. The provider web console
always works regardless and is your lockout-proof fallback.

## 2. Dependencies

```bash
apt update && apt install -y git python3-venv python3-pip libssl-dev libffi-dev build-essential
```

Cowrie needs Python >= 3.11. Ubuntu 22.04 ships 3.10, so add 3.11:

```bash
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version   # expect 3.11.x
```

## 3. Create the cowrie user (never run Cowrie as root)

```bash
useradd -m -s /bin/bash cowrie
id cowrie                # confirm the user exists
su - cowrie
```

## 4. Install Cowrie (build the venv with python3.11, not python3)

```bash
git clone https://github.com/cowrie/cowrie
cd cowrie
python3.11 -m venv cowrie-env
source cowrie-env/bin/activate
pip install --upgrade pip
pip install -e .
```

## 5. Enable JSON logging

The config template location varies by version; copy it into `etc/`:

```bash
cp src/cowrie/data/etc/cowrie.cfg.dist etc/cowrie.cfg   # or etc/cowrie.cfg.dist on older versions
nano etc/cowrie.cfg
```

Find `[output_jsonlog]` (Ctrl+W in nano) and set:

```ini
[output_jsonlog]
enabled = true
logfile = ${honeypot:log_path}/cowrie.json
```

Save and exit. `cowrie.json` is the file the HoneyAI parser consumes.

## 6. Start Cowrie

```bash
cowrie start                 # recent versions; older use bin/cowrie start
ss -tlnp | grep 2222         # confirm it is listening on 2222
```

## 7. Redirect port 22 -> Cowrie (as root)

```bash
exit                         # leave the cowrie user
iptables -t nat -A PREROUTING -p tcp --dport 22 -j REDIRECT --to-port 2222
iptables -t nat -L PREROUTING -n   # confirm the REDIRECT rule
```

Make it survive reboots:

```bash
apt install -y iptables-persistent   # choose <Yes> to save current rules
```

Optional, auto-restart Cowrie after a reboot:

```bash
(crontab -u cowrie -l 2>/dev/null; echo "@reboot cd /home/cowrie/cowrie && /home/cowrie/cowrie/cowrie-env/bin/cowrie start") | crontab -u cowrie -
```

## 8. Security (protect the real host, keep the trap open)

- **Admin SSH (port 2277):** use an SSH key, then set `PasswordAuthentication no`
  in `/etc/ssh/sshd_config` and `systemctl restart ssh`. Keep the web console as a
  fallback so you cannot lock yourself out.
- **Cloud firewall** (provider dashboard): allow inbound **TCP 22** (honeypot) and
  **TCP 2277** (admin) only; block the rest.
- **Patches:** confirm `unattended-upgrades` is active.
- **Do NOT** run fail2ban on port 22 - the brute-force attempts are the data we want.
- Treat the box as disposable: no sensitive data, no reused passwords.

## 9. Collect, then analyse

Let it run at least 1-2 weeks. Check the haul any time:

```bash
su - cowrie
wc -l /home/cowrie/cowrie/var/log/cowrie/cowrie.json
```

To analyse, get `cowrie.json` to the team and run it through the Cowrie parser
(`honeyai/parsers/cowrie.py`, TODO) which maps sessions onto the HoneyAI schema as
unlabelled wild traffic.
