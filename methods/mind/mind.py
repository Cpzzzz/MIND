from datetime import datetime
from methods.mind.optimizer import optimize_with_examples, optimize_without_example
from utils.metrics.evaluator import UnifiedEvaluator
from utils.llm.llm_calling import call_model
from utils.metrics.toxicity import analyze_text_toxicity
from utils.memory.retrieval import retrieve_memory

class MIND:
    @staticmethod
    def _toxicity_profile(tox_scores):
        if not isinstance(tox_scores, dict) or not tox_scores:
            return {
                'max_attr': None,
                'max_score': None,
                'active_attrs': [],
            }
        max_attr, max_score = max(tox_scores.items(), key=lambda item: item[1])
        return {
            'max_attr': max_attr,
            'max_score': max_score,
            'active_attrs': [attr for attr, score in tox_scores.items() if score >= 0.1],
        }

    @staticmethod
    def _calculate_qd(ori_tox_max, upd_tox_max, utility, lambda_val):
        delta_t = max(0, ori_tox_max - upd_tox_max) if (ori_tox_max is not None and upd_tox_max is not None) else 0.0
        return delta_t, lambda_val * delta_t + (1 - lambda_val) * utility

    @staticmethod
    def _select_best_record(records, ori_tox_max, tau):
        if not records:
            return None

        def qd(record):
            value = record.get('Q_d')
            return value if value is not None else -1.0

        reduced = [
            record for record in records
            if record.get('upd_tox_max') is not None
            and ori_tox_max is not None
            and record.get('upd_tox_max') < ori_tox_max
        ]
        detoxed = [record for record in reduced if record.get('upd_tox_max') <= tau]
        if detoxed:
            return max(detoxed, key=qd)
        if reduced:
            return max(reduced, key=qd)
        return min(records, key=lambda record: record.get('upd_tox_max', 1.0))

    @staticmethod
    def mind(args, rets):
        """
        MIND Core: Memory-based Iterative Detoxification
        """
        try:
            print(f"\n[MIND] Optimization Start: '{rets.ori_prompt[:50]}...'")
        except (UnicodeEncodeError, OSError):
            print(f"\n[MIND] Optimization Start: '{rets.ori_prompt[:50].encode('ascii', 'replace').decode()}...'")
        
        # 1. Initialization
        evaluator = UnifiedEvaluator(getattr(args, 'eva_model', None))
        all_records = []
        # ori_embedding removed to save space, retrieved on demand if needed
        
        # 2. Iteration Loop
        for t in range(args.T):
            print(f"\n>>> Iteration {t+1}/{args.T}")
            
            # Retrieve Examples
            examples = []
            if getattr(args, 'memory_system', None):
                try:
                    active_memory = args.memory_system.get_all_active()
                    examples = retrieve_memory(
                        rets.ori_prompt,
                        args.n_examples,
                        active_memory,
                        embedding_model=getattr(args, 'embedding_model', None),
                        alpha=getattr(args, 'mr_alpha', 1.0),
                        beta=getattr(args, 'mr_beta', 1.0),
                        gamma=getattr(args, 'mr_gamma', 1.0),
                        lambda_val=getattr(args, 'lambda_val', 0.5),
                    )
                except Exception as e:
                    print(f"   [Warning] Memory Retrieval Failed: {e}")
                    examples = []
            
            # Calculate Original Toxicity Max (Reference)
            ori_tox_scores = getattr(rets, 'ori_tox', {})
            ori_tox_max = max(ori_tox_scores.values()) if isinstance(ori_tox_scores, dict) and ori_tox_scores else 0.0

            if examples:
                print(f"   Strategy: Optimize with {len(examples)} examples")
                upd_prompt = optimize_with_examples(rets.ori_prompt, examples, args)
            else:
                print("   Strategy: Optimize without examples")
                upd_prompt = optimize_without_example(rets.ori_prompt, args)

            if not upd_prompt:
                raise RuntimeError("MIND failed: optimizer returned no prompt.")

            upd_response = call_model(
                args.llm_model,
                upd_prompt,
                task="[Target] Generate",
            )
            if not upd_response:
                raise RuntimeError("MIND failed: target model returned no response.")

            curr_tox_scores = analyze_text_toxicity(upd_response)
            if not curr_tox_scores:
                raise RuntimeError("MIND failed: toxicity evaluation returned no scores.")

            metrics = evaluator.evaluate(
                rets.ori_prompt,
                upd_response,
                tox_details=curr_tox_scores,
            )
            if metrics.get('tox_max') is None or metrics.get('utility') is None:
                raise RuntimeError("MIND failed: candidate evaluation was incomplete.")

            tox_score = metrics.get('tox_max')
            util_score = metrics.get('utility')
            try:
                print(f"   Result: Tox={tox_score:.4f} | Util={util_score:.4f} | Prompt: {upd_prompt[:40]}...")
            except (UnicodeEncodeError, OSError):
                print(f"   Result: Tox={tox_score:.4f} | Util={util_score:.4f} | Prompt: {upd_prompt[:40].encode('ascii', 'replace').decode()}...")

            ori_tox_avg = sum(ori_tox_scores.values()) / len(ori_tox_scores) if isinstance(ori_tox_scores, dict) and ori_tox_scores else 0.0
            upd_tox_scores = metrics.get('tox_details', {})
            upd_tox_max = metrics.get('tox_max')
            upd_tox_avg = metrics.get('tox_avg')
            delta_t, q_d = MIND._calculate_qd(ori_tox_max, upd_tox_max, util_score, getattr(args, 'lambda_val', 0.5))

            iteration_best = {
                'ori_prompt': rets.ori_prompt,
                'ori_response': rets.ori_response,
                'ori_tox': ori_tox_scores,
                'ori_tox_max': ori_tox_max,
                'ori_tox_avg': ori_tox_avg,
                'ori_toxicity_profile': MIND._toxicity_profile(ori_tox_scores),
                'upd_prompt': upd_prompt,
                'upd_response': upd_response,
                'upd_tox': upd_tox_scores,
                'upd_tox_max': upd_tox_max,
                'upd_tox_avg': upd_tox_avg,
                'upd_toxicity_profile': MIND._toxicity_profile(upd_tox_scores),
                'intent_preservation': metrics.get('intent_preservation'),
                'helpfulness': metrics.get('helpfulness'),
                'informativeness': metrics.get('informativeness'),
                'utility': util_score,
                'Q_d': q_d,
                'delta_tox': delta_t,
                'method': 'MIND',
                'target_model': getattr(args, 'llm_model', None),
                'optimizer_model': getattr(args, 'opt_model', None),
                'judge_model': getattr(args, 'eva_model', None),
                'embedding_model': getattr(args, 'embedding_model', None),
                'iteration': t + 1,
                'mr_alpha': getattr(args, 'mr_alpha', None),
                'mr_beta': getattr(args, 'mr_beta', None),
                'mr_gamma': getattr(args, 'mr_gamma', None),
                'mm_alpha': getattr(args, 'mm_alpha', None),
                'mm_beta': getattr(args, 'mm_beta', None),
                'lambda_val': getattr(args, 'lambda_val', None),
                'timestamp': datetime.now().isoformat()
            }
            all_records.append(iteration_best)
            print(
                f"   [Selected] Iteration best Qd={iteration_best.get('Q_d'):.4f} "
                f"Tox={iteration_best.get('upd_tox_max'):.4f}"
            )

            memory_saved = iteration_best.get('upd_tox_max') is not None and iteration_best.get('upd_tox_max') < ori_tox_max
            iteration_best['memory_saved'] = memory_saved
            iteration_best['memory_skip_reason'] = None if memory_saved else 'toxicity_not_reduced'

            if memory_saved and getattr(args, 'memory_system', None):
                try:
                    args.memory_system.add_to_history([iteration_best])
                    args.memory_system.update_active_memory([iteration_best])
                except Exception as e:
                    iteration_best['memory_saved'] = False
                    iteration_best['memory_skip_reason'] = 'memory_update_failed'
                    print(f"   [Error] Memory Update Failed: {e}")
            elif not memory_saved:
                print("   [Memory] Skipped: toxicity was not reduced.")

        # 3. Finalize
        if all_records:
            ori_tox_scores = getattr(rets, 'ori_tox', {})
            ori_tox_max = max(ori_tox_scores.values()) if isinstance(ori_tox_scores, dict) and ori_tox_scores else 0.0
            best = MIND._select_best_record(all_records, ori_tox_max, getattr(args, 'tau', 0.1))
            vars(rets).update({
                'result': 'Completed', 
                'upd_prompt': best['upd_prompt'], 
                'upd_response': best['upd_response'], 
                'upd_tox_max': best.get('upd_tox_max'),
                'best_record': best, 
                'all_records': all_records
            })
            print(f"\n[MIND] Completed. Best Qd: {best.get('Q_d'):.4f} | Toxicity: {best.get('upd_tox_max'):.4f}")
        else:
            rets.result = 'Error'
            print("\n[MIND] Failed: No valid records.")
