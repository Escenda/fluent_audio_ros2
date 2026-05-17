# Correction Processing

This category contains nodes that repair or clean input audio.

Examples:

- denoise
- acoustic echo cancellation
- dereverberation
- declip
- declick/decrackle
- debreath
- wind noise reduction
- hum removal
- DC offset removal

Correction nodes must expose their model/runtime requirements explicitly. A
missing model, missing reference signal, or unsupported runtime must not be
silently replaced with a weaker path.
