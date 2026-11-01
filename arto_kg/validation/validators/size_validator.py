   
            

"""Size validation: checks detected box areas against declared size labels."""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import logging

from .. import size_ordering as so
from .. import instance_assignment as ia

logger = logging.getLogger(__name__)


class SizeValidator:
    """Object size relation validator"""
    
    def extract_expected_size_relations(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect the declared size labels, keyed by instance_id.

        instance_id matches the bounding-box keys, so same-named objects stay
        separate. Only the `size` each object declares is used: an object
        without one has no size expectation and is not scored.
        """
        size_relations = {
            'absolute_sizes': {},
            'object_names': {},   # instance_id -> name, for reporting
        }

        for instance_id, obj in ia.iter_objects(json_data):
            obj_name = obj.get('name', '')
            if not obj_name:
                continue
            size_relations['object_names'][instance_id] = obj_name
            if obj.get('size'):
                size_relations['absolute_sizes'][instance_id] = obj['size']

        return size_relations

    def analyze_absolute_sizes(self, bboxes_dict: Dict[str, List[int]],
                             expected_sizes: Dict[str, str]) -> Dict[str, Any]:
        """Check that detected areas follow the declared size ordering.

        Size labels (Dominant, Large, Medium, Small, Marginal) are ordinal, so
        the score is rank agreement between declared labels and box areas.
        """
        if not expected_sizes:
            return {'score': None, 'note': 'No absolute size expectations',
                    'objects_analyzed': 0}

        try:
            objects = [
                {'instance_id': name, 'size': size, 'bbox': bboxes_dict.get(name)}
                for name, size in expected_sizes.items()
            ]
            result = so.evaluate(objects)

            return {
                # None when no pair could be compared.
                'score': result['score'],
                'method': 'rank_agreement',
                'pairs_compared': result['pairs'],
                'pairs_concordant': result['concordant'],
                'detailed_analysis': result['details'],
                'skipped': result['skipped'],
                'objects_analyzed': len(objects) - len(result['skipped']),
            }

        except Exception as e:
            logger.error(f"Absolute size analysis failed: {e}")
            return {'score': None, 'error': str(e)}
    
    def create_size_visualization(self, evaluation_result: Dict[str, Any], 
                                save_path: Optional[str] = None) -> Optional[str]:
        """Plot the size-ordering result; save to `save_path` or show it."""
        try:
            analysis = evaluation_result.get('size_analysis', {})
            score = evaluation_result.get('overall_score')
            pairs = analysis.get('pairs_compared') or 0
            concordant = analysis.get('pairs_concordant') or 0

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle('Size Ordering (score: ' + ('-' if score is None else f'{score:.1f}/100') + ')',
                         fontsize=16, fontweight='bold')

            # 1. Pairs whose detected areas agree or disagree with the declared order
            if pairs:
                axes[0].bar(['concordant', 'discordant'], [concordant, pairs - concordant],
                            color=['green', 'red'], alpha=0.7)
                axes[0].set_ylabel('Pairs')
            else:
                axes[0].text(0.5, 0.5, 'No comparable pairs', ha='center', va='center',
                             transform=axes[0].transAxes)
            axes[0].set_title('Declared order vs detected areas')

            # 2. Summary, with the discordant pairs listed
            discordant = [p for p in analysis.get('detailed_analysis') or [] if not p.get('concordant')]
            lines = [f"Objects analysed: {analysis.get('objects_analyzed', 0)}",
                     f"Objects skipped: {len(analysis.get('skipped') or [])}",
                     f"Pairs compared: {pairs}",
                     f"Pairs concordant: {concordant}"]
            if discordant:
                lines += ["", "Discordant (declared larger < smaller):"]
                lines += [f"  {p['larger_declared']} < {p['smaller_declared']}" for p in discordant[:10]]
            axes[1].text(0.05, 0.95, "\n".join(lines), transform=axes[1].transAxes,
                         fontsize=11, verticalalignment='top', family='monospace')
            axes[1].set_title('Summary')
            axes[1].axis('off')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Size visualization failed: {e}")
            if save_path:
                # Save a placeholder image with the error message
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f"Visualization failed:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                plt.savefig(save_path, dpi=150)
                plt.close()
                return save_path
            return None
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str, 
                       bboxes_dict: Dict[str, List[int]],
                       visualize: bool = False) -> Dict[str, Any]:
        """Score the declared size ordering against the detected boxes."""
        try:
            logger.info(f"Starting size validation for: {image_path}")
            
            # 1. Extract expected size relations
            expected_size_relations = self.extract_expected_size_relations(json_data)

            # 2. Score the declared ordering against the detected areas.
            # Pairwise size phrases derive from the same labels, so they are
            # not scored separately.
            analysis = self.analyze_absolute_sizes(
                bboxes_dict, expected_size_relations.get('absolute_sizes', {})
            )
            overall_score = analysis.get('score')

            # 3. Assemble the result
            result = {
                'overall_score': overall_score,
                'method': 'rank_agreement',
                'size_analysis': analysis,
                'evaluation_summary': {
                    'total_objects': len(bboxes_dict),
                    'objects_with_size_expectations': len(expected_size_relations.get('absolute_sizes', {})),
                    'expected_relations': expected_size_relations
                },
                'image_path': image_path,
                'visualization_path': None
            }

            # 4. Visualization, drawn from the assembled result
            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "size_analysis")
                os.makedirs(output_dir, exist_ok=True)
                viz_path = os.path.join(output_dir, f"size_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                result['visualization_path'] = self.create_size_visualization(result, viz_path)
            
            # A None score means no pair could be compared; log it without
            # numeric formatting.
            if overall_score is None:
                logger.info("Size validation completed. No size expectation to check.")
            else:
                logger.info(f"Size validation completed. Overall score: {overall_score:.1f}/100")

            return result

        except Exception as e:
            logger.error(f"Size validation failed: {e}")
            return {
                'error': str(e),
                # None: a failed run measured nothing.
                'overall_score': None,
                'image_path': image_path
            }
