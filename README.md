# Simonides

**A private, long-term memory for your AI assistant, kept on your own computer.**

Out of the box, an AI assistant like Claude forgets you when the conversation
ends. Simonides gives it a place to write things down and look them up again
later: notes, facts about people and projects, and its own diary. You talk to
your assistant normally; it decides what to file and finds it again when it
needs it.

Everything stays on your computer, locked with encryption. Nothing is sent to a
company's servers. It is free, and it will stay free: the licence (AGPL-3.0)
means nobody can take it, close it up and sell it back to you.

The name comes from Simonides of Ceos, the Greek poet credited with inventing
the "memory palace": remembering things by placing them in the rooms of an
imagined building. Simonides works the same way. Notes go into **rooms**, rooms
sit in **wings**, and each note is a **drawer**.

> **Version 0.1.** It works, and it has been tested and reviewed, but it is
> new. Read [What it can't do yet](#what-it-cant-do-yet) before you trust it
> with anything you can't afford to lose.

---

## Setting it up, step by step

This takes about half an hour, most of it waiting. You type a few commands into
a terminal window. Copy them exactly; you don't need to understand them.

### Step 1. Install the tools Simonides is built with

You do this once. Simonides is written in a programming language called Rust,
so your computer needs Rust and a few helpers to build it.

**On Windows**

1. Install **Git**: download it from <https://git-scm.com/download/win> and
   click through the installer, accepting the defaults.
2. Install **Visual Studio Build Tools**: download them from
   <https://visualstudio.microsoft.com/visual-cpp-build-tools/>. When the
   installer asks what to install, tick **"Desktop development with C++"**,
   then install. This one is big; let it run.
3. Install **Strawberry Perl**: download it from <https://strawberryperl.com>
   and accept the defaults.
4. Install **Rust**: download `rustup-init.exe` from <https://rustup.rs>, run
   it, and press Enter to accept the defaults.
5. Restart your computer so it notices everything you just installed.

**On a Mac**

1. Open **Terminal** (press Command + Space, type `Terminal`, press Enter).
2. Paste this, press Enter, and click **Install** in the window that appears:

   ```sh
   xcode-select --install
   ```

3. When that has finished, paste this and press Enter, then press Enter again
   to accept the defaults:

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. Close Terminal and open it again.

**On Linux (Ubuntu or Debian)**

```sh
sudo apt install build-essential perl git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then close the terminal and open a new one.

### Step 2. Download and build Simonides

Open a terminal: on Windows, open the Start menu and type `PowerShell`; on a
Mac, open **Terminal**. Paste these three lines, pressing Enter after each:

```sh
git clone https://github.com/runsthenumbers/simonides.git
cd simonides
cargo install --path crates/simonides-mcp
```

The last one builds Simonides. **The first time takes 10 to 20 minutes** and
prints a lot of text. That is normal. It is done when you get your prompt back
and the last lines say `Installed` without the word `error`.

### Step 3. Set it up

Close the terminal, open a new one, and type:

```sh
simonides --setup
```

This creates your memory palace and a secret key that unlocks it. It prints two
things you need:

- where your palace lives (a folder on your computer), and
- a short block of text that starts with `{` and ends with `}`. Leave the
  window open; you will copy that block in the next step.

> **Important: back up your key.** Inside that folder is a file called `key`.
> It is the only way to open your palace. If you lose it, nobody can get your
> memories back, not even us. That is the point of the encryption. Copy the
> `key` file somewhere safe, such as a USB stick or your password manager.

### Step 4. Connect it to Claude Desktop

1. Open **Claude Desktop**.
2. Open **Settings**, then **Developer**, then click **Edit Config**. A file
   called `claude_desktop_config.json` opens (or your computer shows you where
   it is; open it with Notepad or TextEdit).
3. If the file is empty or only contains `{}`, replace everything in it with
   the block that `simonides --setup` printed. If it already has an
   `"mcpServers"` section, ask Claude to help you merge the two; a missing
   comma or bracket will stop it working.
4. Save the file, **quit Claude Desktop completely**, and open it again.

You are done. To check, ask Claude:

> *"Please remember that my sister's birthday is the 3rd of March."*

Then start a **new** conversation and ask:

> *"When is my sister's birthday?"*

If Claude finds it, your assistant has a memory.

The first time it searches, Simonides downloads a small language model (about
130 MB) that lets it find things by meaning, not just exact words. That takes a
minute once; after that, everything happens on your computer.

**Using Claude Code or Cursor instead?** See
[Other assistants](#other-assistants) below.

---

## If something goes wrong

- **"simonides is not recognized" or "command not found".** Close the terminal
  and open a new one. If that doesn't help, restart the computer; the Rust
  installer adds Simonides' folder to your path, and some systems only notice
  after a restart.
- **The build in Step 2 stops with an error mentioning `perl` or `openssl`.**
  On Windows, check that Strawberry Perl is installed (Step 1.3) and restart.
  On Linux, run `sudo apt install perl`.
- **The build stops with an error mentioning `link.exe` or `cc`.** The C++
  build tools are missing: on Windows, repeat Step 1.2 and make sure "Desktop
  development with C++" is ticked; on a Mac, repeat `xcode-select --install`.
- **Claude doesn't seem to use its memory.** Make sure you quit Claude Desktop
  completely (on Windows, check the icons by the clock too) and reopened it.
  Then ask it directly: *"Do you have the Simonides tools?"*
- **Something else.** Open an issue on this page (the **Issues** tab at the
  top) and paste what the terminal said.

---

## What it can't do yet

Stated plainly, so you can decide what to trust it with:

- **One computer, one person.** Your palace lives on one machine. Keeping it
  in step across several devices is not available yet; the groundwork is in the
  code but is experimental and not switched on.
- **The key file is the lock.** It is protected so that other accounts on your
  computer can't read it, but anyone who can log in as you can open your
  palace.
- **The "vault" tier can't be opened yet.** If you use the optional
  three-tier setup described below, anything routed to the vault is refused
  rather than stored, because this version has no way to unlock it.
- **No independent security audit** has been done. It has been reviewed and
  tested, but not by an outside security firm.

---

# For developers

Everything below is technical detail.

## Requirements

Rust 1.82 or newer (tested with 1.95 and 1.96) and a C toolchain. The build
compiles SQLCipher and OpenSSL from source, so no system OpenSSL is needed.
Linux: `build-essential` and `perl`. macOS: the Xcode command-line tools.
Windows: Visual Studio Build Tools with the C++ workload, and Strawberry Perl
(the Perl bundled with Git for Windows will not build OpenSSL).

`cargo install --path crates/simonides-mcp` puts the `simonides` binary in
Cargo's bin folder. `simonides --help` lists every option.

## Where things live

| OS      | Home folder                                  |
|---------|----------------------------------------------|
| Windows | `%APPDATA%\simonides`                        |
| macOS   | `~/Library/Application Support/simonides`    |
| Linux   | `~/.local/share/simonides`                   |

The home folder holds the encrypted database (`palace.db`), encrypted blobs
(`vault/`), the 256-bit database key (`key`, readable only by your account:
mode 0600 on Unix, an owner-only ACL on Windows) and the embedding model cache
(`models/`). Search is hybrid: BM25 full-text plus local vector embeddings
(BAAI bge-small-en-v1.5), fused with reciprocal rank fusion. New drawers are
embedded in the background; until then they match by keyword.

## Other assistants

Simonides speaks MCP over stdin and stdout, so the client launches it. Use the
full path to the binary; `simonides --setup` prints it.

**Claude Code:**

```sh
claude mcp add --scope user simonides -- /full/path/to/simonides
```

**Cursor:** add this to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "simonides": {
      "command": "/full/path/to/simonides"
    }
  }
}
```

In JSON on Windows, write the path with doubled backslashes, for example
`"C:\\Users\\you\\.cargo\\bin\\simonides.exe"`. Any MCP client that can launch
a stdio server works the same way.
## What the agent gets

The tool surface is modelled on MemPalace's,
so agents and prompts written for it carry over:

- **Drawers** (notes): `add_drawer`, `get_drawer`, `update_drawer`,
  `delete_drawer`, `list_drawers`, `check_duplicate`, `search`
- **Structure:** `list_wings`, `list_rooms`, `get_taxonomy`, `status`
- **Links:** `create_tunnel`, `list_tunnels`, `find_tunnels`, `follow_tunnels`,
  `delete_tunnel`, `traverse`, `graph_stats`
- **Knowledge graph:** `kg_add`, `kg_invalidate`, `kg_query`, `kg_stats`,
  `kg_timeline`
- **Diaries:** `diary_write`, `diary_read`
- **Maintenance:** `rebuild`, `rebuild_status`, `membrane_status`, and the
  MemPalace compatibility tools `get_aaak_spec`, `memories_filed_away`,
  `hook_settings`, `reconnect`

No tool accepts a key or passphrase. Secrets come only from your machine.

## Configuration

Everything is optional. `simonides --help` lists it all.

| Variable                 | Meaning                                         | Default             |
|--------------------------|-------------------------------------------------|---------------------|
| `SIMONIDES_HOME`         | Folder for store, key and model cache           | per-OS, above       |
| `SIMONIDES_KEY`          | Database key, instead of the key file           | read `<home>/key`   |
| `SIMONIDES_DB`           | Database file                                   | `<home>/palace.db`  |
| `SIMONIDES_VAULT`        | Blob folder                                     | `<home>/vault`      |
| `SIMONIDES_TIER_POLICY`  | Policy file; turns on the three tiers (below)   | unset               |
| `SIMONIDES_EMBED_BATCH`  | Chunks embedded per worker process              | `2000`              |
| `FASTEMBED_CACHE_DIR`    | Embedding model cache                           | `<home>/models`     |

If you set `SIMONIDES_KEY` in a client configuration file, that file now holds
your key. Prefer the key file.

## Three tiers (optional)

Simonides can split one palace into three encrypted stores, so that what an
agent may share is kept apart from what must stay on the machine:

- **L1, open.** The only tier that may ever be served beyond this machine.
- **L2, private.** Encrypted under its own key, kept on this machine.
- **L3, vault.** Always starts locked. In this release it has no unlock path,
  so drawers routed to L3 are refused rather than written.

A policy file decides which tier each wing and room belongs in. Rules are
checked in order, the first match wins, and anything no rule matches goes to
the default, which is L3: a private note misfiled into an open tier has already
leaked, while an over-sealed one is only a nuisance. Start from
[examples/tier-policy.toml](examples/tier-policy.toml), then:

```json
{
  "mcpServers": {
    "simonides": {
      "command": "/full/path/to/simonides",
      "env": { "SIMONIDES_TIER_POLICY": "/full/path/to/tier-policy.toml" }
    }
  }
}
```

The L2 key is generated into `<home>/key-l2` on first run (or set
`SIMONIDES_L2_KEY`). There is deliberately no way to give the L3 key through the
environment; setting `SIMONIDES_L3_KEY` stops the server.

## Limitations

Stated plainly, so you can decide what to trust it with:

- **Key derivation parameters are provisional.** Passphrase-derived keys use
  Argon2id at OWASP's recommended minimum (19 MiB, 2 passes). They have not been
  tuned per device. The default single store does not use a passphrase; it uses
  a random 256-bit key from the key file.
- **L3 cannot be unlocked in this release.** The vault tier exists and is
  enforced, but there is no unlock command yet.
- **Sync is experimental.** The crates include CRDT merge rules for drawers,
  the knowledge graph and palace structure, tested for convergence within one
  process. There is no network transport, deltas are full snapshots, and
  multi-device use has not been proven. Do not rely on it.
- **Single user, single machine.** The key file is protected by file
  permissions (mode 0600 on Unix; an owner-only ACL on Windows). Anyone who can
  read your files as you can read your palace.
- **No independent security audit** has been done.

## Building and testing

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The workspace has five crates: `simonides-core` (shared types),
`simonides-crypto` (key derivation and AEAD primitives), `simonides-sync`
(merge rules), `simonides` (the store, retrieval, embeddings, knowledge graph
and tiers) and `simonides-mcp` (the server binary).

## Third-party licences

Dependencies are fetched by Cargo and keep their own licences. All are
compatible with the AGPL-3.0: most are MIT or Apache-2.0, a few are BSD, ISC,
Zlib, Unicode-3.0 or CDLA-Permissive-2.0, and four are MPL-2.0 (`im`,
`bitmaps` and `sized-chunks` through Automerge, `option-ext` through `dirs`).
The build compiles SQLCipher (BSD-style, Zetetic LLC) and OpenSSL 3
(Apache-2.0) from source, and links ONNX Runtime (MIT), which the `ort` crate
downloads prebuilt at build time. The embedding model,
BAAI bge-small-en-v1.5, is MIT-licensed and is downloaded at run time, not
shipped. If you distribute binaries, include those projects' notices.

## Credits

Written by Matthew Attenborough and Minerva. Minerva is an AI model instance that
designed and wrote most of the code. Copyright is held by Matthew Attenborough.

The data model and tool surface follow MemPalace (MIT License); see
[NOTICE](NOTICE).