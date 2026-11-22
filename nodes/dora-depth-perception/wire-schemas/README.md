# wire-schemas

The Arrow type of every message `lib/dora-rig-messages` defines, under its stable name, and each type's fingerprint.
It depends on `arrow-schema` (the version dora-node-api's arrow uses, so the types are the node API's own) and `sha2`
only, so dora-zenoh-bridge can use it without perception-wire's dependencies.

- Each module (`perception`, `camera`, `localization`, `navigation`, `objects`, `odometry`, `people`, `posed_depth`,
  `space_map`, `robot`) gives the `Fields` of its messages and of the structs inside them. perception-wire encodes and
  decodes with these; dora-serial-lidar writes its `scan` with them.
- `WIRE_TYPES` lists every message type by name; `WireTypeTable` finds one by fingerprint or by Arrow type. Messages
  of one Arrow type share an entry: the bridge cannot tell them apart and need not, since it decodes them alike.
- `fingerprint` is the first 8 bytes of the SHA-256 of `canonical_encoding`, an ASCII spelling of the whole type
  (field names, types, nullability, nesting, in order; the grammar is in `src/fingerprint.rs`).
  `dora_rig_messages.wire` spells the same encoding in Python.

The tests compare every type's fingerprint and canonical encoding with `fixtures/wire_fingerprints.json`, which
`python -m dora_rig_messages.fixtures` writes from the Python definitions, so a field changed in one language only is
a failing test. A change to a message goes in both languages, then the fixtures are regenerated.
