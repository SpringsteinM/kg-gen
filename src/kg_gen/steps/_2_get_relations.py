import dspy
import warnings

from pydantic import BaseModel, create_model
from pydantic_core import ValidationError
from typing import List, Literal, Optional, Dict, Tuple


class Relation(BaseModel):
  "Knowledge graph subject-predicate-object tuple"
  subject: str
  predicate: str
  object: str
    

class RelationWithType(Relation):
  "Knowledge graph subject-predicate-object tuple with predicate type"
  predicate_type: Optional[str] = None


class _ExtractBase(dspy.Signature):
  source_text: str = dspy.InputField()
  entities: List[str] = dspy.InputField()


class ExtractTextRelations(_ExtractBase):
  """Extract subject-predicate-object triples from the source text. Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
      
  relations: list[dict] = dspy.OutputField(desc="List of subject-predicate-object tuples. Be thorough")


class ExtractConversationRelations(_ExtractBase):
  """Extract subject-predicate-object triples from the conversation, including:
  1. Relations between concepts discussed
  2. Relations between speakers and concepts (e.g. user asks about X)
  3. Relations between speakers (e.g. assistant responds to user)
  Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
      
  relations: list[dict] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough")


class _ExtractBaseWithTypes(_ExtractBase):
  edge_types: List[str] = dspy.InputField(desc="List of allowed predicate types")
  require_type: bool = dspy.InputField(desc="Whether every predicate must have a type from edge_types")
  relations: list[dict] = dspy.OutputField(desc="List of subject-predicate-object tuples with predicate types. Be thorough")


class ExtractTextRelationsWithTypes(_ExtractBaseWithTypes):
  """Extract subject-predicate-object triples from the source text and assign types to predicates. Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


class ExtractConversationRelationsWithTypes(_ExtractBaseWithTypes):
  """Extract subject-predicate-object triples from the conversation and assign types to predicates. Include relations between concepts, speakers and concepts, and between speakers. Subject and object must be from entities list.
  This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""


def get_relations(
  dspy: dspy.dspy,
  input_data: str,
  entities: list[str],
  is_conversation: bool = False,
  edge_types: Optional[List[str]] = None,
  require_edge_type: bool = True,
  examples: Optional[List[dspy.Example]] = None,
) -> Tuple[List[Tuple[str, str, str]], Optional[Dict[str, str]]]:
  """Extract relations from input data.
  
  Args:
      dspy: DSPy module
      input_data: Text to process
      entities: List of entities to relate
      is_conversation: Whether the input is a conversation
      edge_types: Optional list of allowed edge types
      require_edge_type: Whether every edge must have a type from edge_types
      examples: Optional list of examples to use as in‑context demonstrations
      
  Returns:
      Tuple of (list of relation tuples, dictionary mapping predicates to types)
  """
  if edge_types:
    Signature = (
      ExtractConversationRelationsWithTypes
      if is_conversation
      else ExtractTextRelationsWithTypes
    )

    params = {
      "source_text": input_data,
      "entities": entities,
      "edge_types": edge_types,
      "require_type": require_edge_type,
    }
  else:
    Signature = (
      ExtractConversationRelations
      if is_conversation
      else ExtractTextRelations
    )

    params = {
      "source_text": input_data,
      "entities": entities,
    }

  if examples:
    params["demos"] = [
      dspy.Example(
        source_text=example.source_text,
        relations=example.relations,
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
    
  relations = []
  edge_type_map = {}

  for item in result.relations:
    try:
      if edge_types:
        rel = RelationWithType(**item)
      else:
        rel = Relation(**item)
    except ValidationError as e:
      warnings.warn(
        f"Invalid relation {item!r}: {e}",
        category=UserWarning,
        stacklevel=2,
      )
      continue
    except Exception as e:
        # TODO
        continue

    if rel.subject in entities and rel.object in entities:
      relations.append((rel.subject, rel.predicate, rel.object))
      if edge_types and rel.predicate_type:
        edge_type_map[rel.predicate] = rel.predicate_type

  return relations, (edge_type_map if edge_types else None)
