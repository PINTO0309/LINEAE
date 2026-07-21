import torch
import torch.distributed as dist

SAP_EVALUATION_PROTOCOL = 'official_all_queries_and_deployment_topk'

__all__ = ['DualLineEvaluator', 'LineEvaluator', 'SAP_EVALUATION_PROTOCOL']


class LineEvaluator(object):
    def __init__(self, max_predictions=None):
        if max_predictions is not None and int(max_predictions) <= 0:
            raise ValueError('max_predictions must be positive')
        self.max_predictions = (
            None if max_predictions is None else int(max_predictions)
        )
        self.n_gt = []
        self.distances = []
        self.choices = []
        self.spa_metrics = {'sap5': 5, 'sap10': 10, 'sap15': 15}
        self.tp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.fp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.scores = []

    def prepare(self, lines, scores=None):
        lines = lines.unflatten(-1, (2, 2)).flip([-1])

        if scores is not None:
            scores = scores[..., 0].sigmoid()
            scores_idx = torch.argsort(scores, descending=True, dim=-1)
            if self.max_predictions is not None:
                scores_idx = scores_idx[:, :self.max_predictions]
            scores = torch.gather(scores, 1, scores_idx) 
            lines = torch.gather(lines, 1, scores_idx[:, :, None, None].repeat(1, 1, 2, 2))
            return lines * 128., scores

        return lines * 128.

    def compute_distances(self, pred_lines, gt_lines):
        dist =  ((pred_lines[:, None, :, None] - gt_lines[:, None]) ** 2).sum(-1)
        dist = torch.minimum(
            dist[:, :, 0, 0] + dist[:, :, 1, 1], 
            dist[:, :, 0, 1] + dist[:, :, 1, 0]
            )

        dist, choice = torch.min(dist, 1)
        return dist, choice

    def msTPFP(self, distances, choice, threshold):
        hit = torch.zeros_like(distances)
        tp = torch.zeros_like(distances)
        fp = torch.zeros_like(distances)

        for i in range(len(distances)):
            if distances[i] < threshold and not hit[choice[i]]:
                hit[choice[i]] = True
                tp[i] = 1
            else:
                fp[i] = 1
        return tp, fp

    def update(self, predictions, ground_truth):
        pred_lines, pred_scores = self.prepare(predictions['pred_lines'], predictions['pred_logits'])
        self.scores.append(pred_scores.flatten(0, 1).detach().cpu())

        for pred_l, gt in zip(pred_lines, ground_truth):
            gt_l = self.prepare(gt['lines'])
            self.n_gt.append(len(gt_l))

            if len(gt_l) == 0:
                distances = torch.full(
                    (len(pred_l),), float('inf'), dtype=pred_l.dtype, device=pred_l.device
                )
                choices = torch.zeros(len(pred_l), dtype=torch.long, device=pred_l.device)
            else:
                distances, choices = self.compute_distances(pred_l, gt_l)
            for k, v in self.spa_metrics.items():
                tp, fp = self.msTPFP(distances, choices, v)
                self.tp[k].append(tp.detach().cpu())
                self.fp[k].append(fp.detach().cpu())

    def synchronize_between_processes(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        local = {
            'n_gt': self.n_gt,
            'scores': self.scores,
            'tp': self.tp,
            'fp': self.fp,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        self.n_gt = []
        self.scores = []
        self.tp = {key: [] for key in self.spa_metrics}
        self.fp = {key: [] for key in self.spa_metrics}
        for state in gathered:
            self.n_gt.extend(state['n_gt'])
            self.scores.extend(state['scores'])
            for key in self.spa_metrics:
                self.tp[key].extend(state['tp'][key])
                self.fp[key].extend(state['fp'][key])

    def ap(self, tp, fp):
        recall = tp
        precision = tp / torch.maximum(tp + fp, torch.tensor(1e-9, dtype=tp.dtype, device=tp.device))

        zero = torch.tensor([0.0], dtype=torch.float, device=tp.device)
        one = torch.tensor([1.0], dtype=torch.float, device=tp.device)
        recall = torch.cat((zero, recall, one))
        precision = torch.cat((zero, precision, zero))

        for i in range(precision.size()[0] - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        idx = torch.where(recall[1:] != recall[:-1])[0]
        return torch.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

    # TODO
    # def ap(self, tp, fp):
    #     recall = tp
    #     precision = tp / torch.maximum(tp + fp, torch.tensor(1e-9, dtype=tp.dtype, device=tp.device))

    #     zero = torch.tensor([[0.0]], dtype=torch.float, device=tp.device).repeat(len(recall), 1)
    #     one = torch.tensor([[1.0]], dtype=torch.float, device=tp.device).repeat(len(recall), 1)
    #     recall = torch.cat((zero, recall, one), dim=1)
    #     precision = torch.cat((zero, precision, zero), dim=1)

    #     for i in range(precision.size()[1] - 1, 0, -1):
    #         precision[:, i - 1] = torch.maximum(precision[:, i - 1], precision[:, i])

    #     idx = torch.where(recall[:, 1:] != recall[:, :-1])[0]

    #     return torch.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])

    def accumulate(self,):
        self.sap_results = {}
        if not self.scores:
            self.sap_results = {key: 0.0 for key in self.spa_metrics}
            return
        scores = torch.cat(self.scores)
        n_gt = sum(self.n_gt)
        if n_gt <= 0:
            self.sap_results = {key: 0.0 for key in self.spa_metrics}
            return

        for k in self.spa_metrics:
            tp_ = torch.cat(self.tp[k])
            fp_ = torch.cat(self.fp[k])

            index = torch.argsort(scores, descending=True)
            tp_ = torch.cumsum(tp_[index], dim=0) / n_gt
            fp_ = torch.cumsum(fp_[index], dim=0) / n_gt

            # tp.append(tp_)
            # fp.append(fp_)

        # tp = torch.stack(tp)
        # fp = torch.stack(fp)
        # self.ap(tp, fp)

            self.sap_results[k] = self.ap(tp_, fp_).item() * 100

    def summarize(self,):
        for sap, results in self.sap_results.items():
            print(f'{sap}:\t{results:.1f}')

    def cleanup(self, ):
        del self.n_gt, self.tp, self.fp, self.scores
        torch.cuda.empty_cache()
        self.n_gt = []
        self.tp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.fp = {'sap5': [], 'sap10': [], 'sap15': []}
        self.scores = []


class DualLineEvaluator(object):
    """Report official all-query and deployment top-k sAP in one pass."""

    def __init__(self, deploy_max_predictions):
        if deploy_max_predictions is None or int(deploy_max_predictions) <= 0:
            raise ValueError('deploy_max_predictions must be positive')
        self.deploy_max_predictions = int(deploy_max_predictions)
        self.official = LineEvaluator(max_predictions=None)
        self.deploy = LineEvaluator(max_predictions=self.deploy_max_predictions)
        self.query_count = None
        self.sap_results = {}

    def update(self, predictions, ground_truth):
        query_count = int(predictions['pred_lines'].shape[1])
        if self.query_count is None:
            self.query_count = query_count
        elif query_count != self.query_count:
            raise ValueError(
                f'inconsistent evaluator query count: {query_count} != {self.query_count}'
            )
        pred_lines, pred_scores = self.official.prepare(
            predictions['pred_lines'], predictions['pred_logits']
        )
        deploy_count = min(self.deploy_max_predictions, query_count)
        self.official.scores.append(pred_scores.flatten(0, 1).detach().cpu())
        self.deploy.scores.append(
            pred_scores[:, :deploy_count].flatten(0, 1).detach().cpu()
        )

        for pred_l, gt in zip(pred_lines, ground_truth):
            gt_l = self.official.prepare(gt['lines'])
            ground_truth_count = len(gt_l)
            self.official.n_gt.append(ground_truth_count)
            self.deploy.n_gt.append(ground_truth_count)
            if ground_truth_count == 0:
                distances = torch.full(
                    (len(pred_l),),
                    float('inf'),
                    dtype=pred_l.dtype,
                    device=pred_l.device,
                )
                choices = torch.zeros(
                    len(pred_l), dtype=torch.long, device=pred_l.device
                )
            else:
                distances, choices = self.official.compute_distances(pred_l, gt_l)
            for metric, threshold in self.official.spa_metrics.items():
                tp, fp = self.official.msTPFP(distances, choices, threshold)
                self.official.tp[metric].append(tp.detach().cpu())
                self.official.fp[metric].append(fp.detach().cpu())
                self.deploy.tp[metric].append(tp[:deploy_count].detach().cpu())
                self.deploy.fp[metric].append(fp[:deploy_count].detach().cpu())

    def synchronize_between_processes(self):
        self.official.synchronize_between_processes()
        self.deploy.synchronize_between_processes()

    def accumulate(self):
        self.official.accumulate()
        self.deploy.accumulate()
        self.sap_results = dict(self.official.sap_results)
        self.sap_results.update({
            f'official_{metric}': value
            for metric, value in self.official.sap_results.items()
        })
        self.sap_results.update({
            f'deploy_{metric}': value
            for metric, value in self.deploy.sap_results.items()
        })

    def summarize(self):
        query_label = self.query_count if self.query_count is not None else 'unknown'
        print(f'Official sAP (all {query_label} queries):')
        self.official.summarize()
        deploy_count = (
            min(self.deploy_max_predictions, self.query_count)
            if self.query_count is not None else self.deploy_max_predictions
        )
        print(f'Deployment sAP (top {deploy_count} queries):')
        for metric, result in self.deploy.sap_results.items():
            print(f'deploy_{metric}:\t{result:.1f}')

    def cleanup(self):
        self.official.cleanup()
        self.deploy.cleanup()
        self.query_count = None
        self.sap_results = {}
