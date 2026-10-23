import { app } from "../../scripts/app.js";

let presets;

app.registerExtension({
  name: "nsfw-review.shieldgemma2-policy-preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ShieldGemma2PolicySignals") return;

    if (!presets) {
      const response = await fetch(new URL("./policy_presets.json", import.meta.url));
      if (!response.ok) throw new Error(`Policy presets: HTTP ${response.status}`);
      presets = await response.json();
    }

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalCreated?.apply(this, args);
      const selector = this.widgets?.find((widget) => widget.name === "policy_mode");
      const custom = this.widgets?.find((widget) => widget.name === "custom_policy");
      if (!selector || !custom) return result;

      const preview = document.createElement("textarea");
      preview.readOnly = true;
      preview.setAttribute("aria-label", "Selected policy text (read only)");
      preview.rows = 7;
      preview.style.width = "100%";
      preview.style.minHeight = "145px";
      preview.style.resize = "vertical";
      preview.style.boxSizing = "border-box";

      this.addDOMWidget("policy_preview", "policy_preview", preview, {
        serialize: false,
      });

      const update = () => {
        const selected = presets[selector.value];
        preview.value = selected
          ? `${selected.label}\n${selected.kind === "official" ? "Official Google policy" : "Project draft, not an official platform policy"}\n\n${selected.text}`
          : `Custom policy\n\n${custom.value || "Enter a policy in custom_policy."}`;
      };
      const originalSelectorCallback = selector.callback;
      selector.callback = (...callbackArgs) => {
        const callbackResult = originalSelectorCallback?.apply(selector, callbackArgs);
        update();
        return callbackResult;
      };
      const originalCustomCallback = custom.callback;
      custom.callback = (...callbackArgs) => {
        const callbackResult = originalCustomCallback?.apply(custom, callbackArgs);
        if (selector.value === "custom") update();
        return callbackResult;
      };

      const originalConfigured = this.onConfigure;
      this.onConfigure = function (...configureArgs) {
        const configured = originalConfigured?.apply(this, configureArgs);
        update();
        return configured;
      };
      update();
      return result;
    };
  },
});
