import dspy
import warnings

from pydantic import BaseModel
from pydantic_core import ValidationError
from typing import List, Optional, Dict, Tuple


class EntityWithType(BaseModel):
  entity: str
  type: Optional[str] = None


class _BaseEntities(dspy.Signature):
  source_text: str = dspy.InputField()
  entities: list[dict] = dspy.OutputField(desc="List of entities. Be thorough")


class TextEntities(_BaseEntities):
  """Extract all entities from the source text. Extracted entities are subjects or objects.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


class ConversationEntities(_BaseEntities):
  """Extract all entities from the conversation. Extracted entities are subjects or objects.
  Consider both explicit entities and participants in the conversation.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


class _BaseEntitiesWithTypes(dspy.Signature):
  source_text: str = dspy.InputField()
  node_types: list[str] = dspy.InputField(desc="List of allowed entity types")
  require_type: bool = dspy.InputField(desc="Whether every entity must have a type from node_types")
  entities: list[dict] = dspy.OutputField(desc="List of entities with their types as {entity: str, type: str}. Be thorough")


class TextEntitiesWithTypes(_BaseEntitiesWithTypes):
  """Extract all entities from the source text and assign types to them. Extracted entities are subjects or objects.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


class ConversationEntitiesWithTypes(_BaseEntitiesWithTypes):
  """Extract all entities from the conversation and assign types to them. Extracted entities are subjects or objects. Consider both explicit entities and participants in the conversation.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


def get_entities(
  dspy: dspy.dspy,
  input_data: str,
  is_conversation: bool = False,
  node_types: Optional[List[str]] = None,
  require_node_type: bool = True,
  examples: Optional[List[dspy.Example]] = None,
) -> Tuple[List[str], Optional[Dict[str, str]]]:
  """Extract entities from the input data, with optional few‑shot examples.
  
  Args:
      dspy: DSPy module
      input_data: Text to process
      is_conversation: Whether the input is a conversation
      node_types: Optional list of allowed node types
      require_node_type: Whether every node must have a type from node_types
      examples: Optional list of examples to use as in‑context demonstrations
      
  Returns:
      Tuple of (list of entity strings, dictionary mapping entities to types)
  """
  if node_types:
    Signature = (
      ConversationEntitiesWithTypes
      if is_conversation
      else TextEntitiesWithTypes
    )

    params = {
      "source_text": input_data,
      "node_types": node_types,
      "require_type": require_node_type,
    }
  else:
    Signature = (
      ConversationEntities
      if is_conversation
      else TextEntities
    )

    params = {"source_text": input_data}

  if examples:
    params["demos"] = [
      dspy.Example(
        source_text=example.source_text,
        entities=example.entities,
      )
      for example in examples
    ]

  try:
    extractor = dspy.Predict(Signature)
    result = extractor(**params)
  except Exception as e:
    warnings.warn(
      f"Extraction failed: {e}",
      category=UserWarning,
      stacklevel=2,
    )
    return [], None

  if node_types:
    entities = []
    entity_types = {}
    
    for item in result.entities:
      try:
        entity = item["entity"]
      except ValidationError as e:
        warnings.warn(
          f"Invalid entity {item!r}: {e}",
          category=UserWarning,
          stacklevel=2,
        )
        continue
      except Exception as e:
        # TODO
        continue

      entities.append(entity)
      if item.get("type"):
        entity_types[entity] = item["type"]

    return entities, entity_types

  return result.entities, None
