"""The single cost formula for usage accounting: one formula for every reader.

Comments and docstrings translated to English for review; logic unchanged. This
is the whole module, not an excerpt. The only substitution is the cloud provider
name, which was made generic.

There are four readers: the journal on the "Tokens" tab, the summary cards, the
per-model breakdown and a user's total spend in settings (plus the cost card on
the activity dashboard). The formula has to be one, or the numbers drift apart,
so the computation lives here rather than in each place separately.

The main rule: **a stored cost outranks a computed one**. If a record already
carries a value (`cost_rub` in `llm_usage`, or in the JSON result of a tool), it
is taken as is and nothing is recomputed. Computation happens only where no
stored value exists.

The two kinds of usage are computed differently and from different sources:
  - LLM: tokens x the price list in `analyst.model_pricing` (editable, which is
    why history is recomputed whenever the price list is edited);
  - images: the price per image from the static `image_client` price list (the
    Image API returns neither cost nor balance), plus a surcharge per reference.

The module deliberately touches neither the DB nor the network: the LLM price
list is passed in as an argument (loaded by `admin_storage._load_pricing`) and
the image price list comes from the `image_client` registry. That way the formula
can be checked hermetically.

User-facing wording is not produced here; the Russian text in this file is in
comments only.
"""

from __future__ import annotations

import re
from typing import Any

import image_client

# The feature value used by image generation rows in analyst.llm_usage.
IMAGE_FEATURE = "image"
# The same for video generation: those rows are written by the video/worker.py
# worker once a clip is ready. The price is already stored there (the provider's
# roubles, or an estimate from its price list), so there is nothing to recompute
# here and no need to.
VIDEO_FEATURE = "video"
# The provider of generation rows: the same vendor as the chat, but a different
# API. The vendor name is replaced with a generic one in this sample.
IMAGE_PROVIDER = "cloud-llm"

# A trailing date in a model id: -YYYY-MM-DD, or -YYYYMMDD / -YYYYMM (6-8 digits).
# Alphabetic suffixes (-mini, -preview and the like) are left alone.
_MODEL_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$|-\d{6,8}$")


def normalize_model(model: str | None) -> str:
    """An id from llm_usage into the base id used to match the price list: the
    provider prefix (vendor/...) and the numeric trailing date are stripped."""
    if not model:
        return ""
    base = model.split("/")[-1]
    return _MODEL_DATE_SUFFIX.sub("", base)


def llm_cost_rub(
    provider: str | None,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing: dict[tuple[str, str], dict[str, float]],
) -> float | None:
    """The cost of one LLM call in roubles. Matched by (provider, base model).
    cached_tokens do NOT enter the price. No price list entry -> None (not
    available), never 0."""
    price = pricing.get((provider or "", normalize_model(model)))
    if price is None:
        return None
    return (
        (prompt_tokens or 0) / 1e6 * price["input_price"]
        + (completion_tokens or 0) / 1e6 * price["output_price"]
    )


def image_cost_rub(
    model_id: str | None,
    *,
    with_reference: bool = False,
    reference_count: int | None = None,
) -> float | None:
    """The cost of one image generation: the model base plus the reference surcharge.

    This is the single place where an image price is computed: when a new
    generation is recorded, when old records are estimated, and in the backfill.

    The rules follow the provider's published price list:
      - the model is unknown, or has no price (base_cost_rub=None) -> None; such
        a record is skipped rather than counted as zero;
      - the reference surcharge applies only if a reference was actually used AND
        the model has an input-image price. A dash in the price list means "input
        is not billed separately", and then the price equals the base;
      - the surcharge is charged PER input image (that is how the price list puts
        it: a surcharge for every uploaded reference image), so it is computed as
        `reference_cost_rub * count`.

    `reference_count` is how many images went into the request. `with_reference`
    is kept for callers that only know the fact ("there was a reference") and not
    the number: old journal records carry no count, and there the surcharge is
    charged for one image, which is exactly how those records were created.
    """
    model = image_client.model_by_id(model_id)
    if model is None or model.base_cost_rub is None:
        return None
    if reference_count is None:
        reference_count = 1 if with_reference else 0
    total = model.base_cost_rub
    if reference_count > 0 and model.reference_cost_rub is not None:
        total += model.reference_cost_rub * reference_count
    return total


def row_cost_rub(
    *,
    feature: str | None,
    provider: str | None,
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    stored_cost_rub: Any = None,
    pricing: dict[tuple[str, str], dict[str, float]],
) -> float | None:
    """The cost of one row (or one homogeneous group) of the llm_usage journal.

    The order is exactly this:
      0. video: the stored cost only, otherwise "not available" (see below);
      1. a stored cost exists -> that is the answer;
      2. an image generation row -> the static image price list. The journal has
         no reference flag, so the base price is used: rows written by the
         application always carry a stored cost, and only records that appeared
         outside it ever reach this branch;
      3. otherwise -> tokens x the LLM price list.
    """
    if stored_cost_rub is not None:
        return float(stored_cost_rub)
    if (feature or "") == IMAGE_FEATURE:
        return image_cost_rub(model)
    if (feature or "") == VIDEO_FEATURE:
        # Video with no stored cost comes from a provider that bills in credits.
        # We have no rouble price list for it, and video has no tokens: there is
        # nothing to compute from, so "not available" rather than substituting an
        # LLM price by model name.
        return None
    return llm_cost_rub(provider, model, prompt_tokens, completion_tokens, pricing)


def generation_cost_rub(result: dict[str, Any] | None) -> float | None:
    """The cost taken from the JSON result of the generate_image tool stored in
    chat_messages.results.

    New records carry a ready `cost_rub`; older ones do not, and the price is
    estimated from `model` plus the surcharge for the attached images. The
    backfill computes it along exactly the same path, which is why the estimate in
    settings and the total in the admin area agree to the kopeck.

    The image count comes from `reference_count` (or the length of
    `reference_image_ids`); records from before multi-reference have neither, and
    there only the scalar `reference_image_id` remains, that is, exactly one image.
    """
    if not isinstance(result, dict) or result.get("status") != "ok":
        return None
    stored = result.get("cost_rub")
    if stored is not None:
        return float(stored)
    count = result.get("reference_count")
    if count is None:
        ids = result.get("reference_image_ids")
        if isinstance(ids, list):
            count = len(ids)
        else:
            count = 1 if result.get("reference_image_id") else 0
    return image_cost_rub(result.get("model"), reference_count=int(count))
