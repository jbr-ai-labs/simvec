import time
import os
import hydra
from copy import deepcopy
from tqdm import trange
from typing import NoReturn
from torch.nn import MSELoss
from omegaconf import DictConfig, OmegaConf
from src.negative_sampling import *
from src.knowledge_graph import KnowledgeGraph, DegreeSimilarityKnowledgeGraph, \
    ParzenSimilarityKnowledgeGraph, MSEKnowledgeGraph, MSEParzenKnowledgeGraph, MSEDegreeSimilarityKnowledgeGraph
from src.trivec_model import TriVec, TriVecWeighted, SPTriVecWeighted
from src.losses import NegativeSoftPlusLoss
from src.utils import switch_grad_mode, switch_model_mode
from src.chemical_embeddings import EmbeddingsInitializer, \
    VectorEmbeddingsInitializer
from run_utils import evaluation, evaluation_ranking
from datetime import datetime
from src.node_similarity_estimator import get_edges_degree_score_multiplication


def make_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def run_train_iteration(kg: KnowledgeGraph, model_save_path, cfg: DictConfig, device, cv_iter=None) -> NoReturn:
    args_psi = 0

    if cfg.run_args.use_weights:
        weighted_triples_loader = data.DataLoader(kg.get_all_drug_pairs(),
                                                  batch_size=cfg.params.batch_size,
                                                  shuffle=True)

    # Pos loaders
    train_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('train'))),
        batch_size=cfg.params.batch_size,
        shuffle=True)

    val_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('val'))),
        batch_size=cfg.params.batch_size,
        shuffle=False)

    test_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('test'))),
        batch_size=cfg.params.batch_size,
        shuffle=False)

    # Neg samplers
    if cfg.run_args.neg_sampling_strategy == "bernoulli":
        train_neg_sampler = BernoulliNegSampler(
            np.array(kg.get_data_by_type('train')),
            kg.get_num_of_ent("train"), kg.get_num_of_ent('train'),
            neg_per_pos=cfg.params.num_of_neg_samples)
    elif cfg.run_args.neg_sampling_strategy == "uniform":
        train_neg_sampler = UniformNegSampler(
            kg.get_num_of_ent("train"), kg.get_num_of_ent('train'),
            neg_per_pos=cfg.params.num_of_neg_samples)
    else:
        train_neg_sampler = UniformNegSampler(
            kg.get_num_of_ent("train"), kg.get_num_of_ent('train'),
            neg_per_pos=cfg.params.num_of_neg_samples)

    if cfg.run_args.random_val_neg_sampler:
        val_neg_sampler = UniformNegSampler(
            kg.get_num_of_ent("val"), kg.get_num_of_ent('val'),
            neg_per_pos=cfg.params.num_of_neg_samples)
    else:
        val_neg_sampler = HonestNegSampler(
            kg.get_num_of_ent("val"), kg.get_num_of_ent('val'),
            kg.get_drugs_array(), neg_per_pos=1)

    test_neg_sampler = HonestNegSampler(
        kg.get_num_of_ent("test"), kg.get_num_of_ent('test'),
        kg.get_drugs_array(), neg_per_pos=1)

    # Neg datasets  (need to can regenerate neg examples)
    val_neg_data = NegDataset(
        torch.Tensor(np.array(kg.get_data_by_type('val'))),
        val_neg_sampler)

    test_neg_data = NegDataset(
        torch.Tensor(np.array(kg.get_data_by_type('test'))),
        test_neg_sampler)

    # Neg loaders
    val_neg_loader = data.DataLoader(val_neg_data,
                                     batch_size=cfg.params.batch_size,
                                     shuffle=False)
    test_neg_loader = data.DataLoader(test_neg_data,
                                      batch_size=cfg.params.batch_size,
                                      shuffle=False)

    # Model init
    if cfg.run_args.use_embeddings_init:
        embed_init = VectorEmbeddingsInitializer(
            cfg.data_const.work_dir + cfg['embedding_data'][cfg.run_args.use_embeddings_init]['data'])
    else:
        embed_init = EmbeddingsInitializer()

    if cfg.run_args.use_weights:
        weight_mtr_similarity = torch.tensor(kg.calculate_pairwise_similarity(), device=device)
        if cfg.run_args.similarity == "parzen":
            weight_mtr_edges_degree = None
        else:
            weight_mtr_edges_degree = torch.tensor(
                kg.calculate_nodes_degrees_matrix(get_edges_degree_score_multiplication),
                device=device)

        model = TriVecWeighted(ent_total=kg.get_num_of_ent('train'),
                               rel_total=kg.get_num_of_rel('train'), dim=cfg.params.embed_dim, embedding_init=embed_init,
                               weights_similarity=weight_mtr_similarity, weights_edges_degree=weight_mtr_edges_degree)
    else:
        model = TriVec(ent_total=kg.get_num_of_ent('train'),
                       rel_total=kg.get_num_of_rel('train'),
                       dim=cfg.params.embed_dim,
                       embedding_init=embed_init)

    if cfg.run_args.neg_sampling_strategy == "stay_positive":
        args_psi = cfg.params.stay_positive_reg_psi
        model = SPTriVecWeighted(ent_total=kg.get_num_of_ent('train'),
                                 rel_total=kg.get_num_of_rel('train'), dim=cfg.run_args.embed_dim, embedding_init=embed_init,
                                 weights_similarity=weight_mtr_similarity if cfg.run_args.use_weights else None,
                                 weights_edges_degree=weight_mtr_edges_degree if cfg.run_args.use_weights else None)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.params.learning_rate,
                           amsgrad=True)
    loss_func = NegativeSoftPlusLoss()
    mse_loss = MSELoss()

    # if not os.path.exists(model_save_path):
    #     os.makedirs(model_save_path)
    val_roc = -1.
    max_val_roc = val_roc
    best_model_params = deepcopy(model.state_dict())
    max_epoch = cfg.params.epoch - 1
    print('Train')
    print()

    early_stop_counter = 0

    for epoch in trange(cfg.params.epoch):
        train_epoch_losses = []
        for pos_triplets in train_pos_loader:
            switch_grad_mode(model, requires_grad=True)
            switch_model_mode(model, train=True)
            pos_triplets = pos_triplets.to(device).long()
            pos_scores = model(pos_triplets) + model.regularization(pos_triplets)
            if cfg.run_args.neg_sampling_strategy != 'stay_positive':
                neg_triplets = train_neg_sampler(pos_triplets).to(device).long()
                neg_scores = model(neg_triplets) + model.regularization(neg_triplets)
                loss = loss_func(pos_scores, neg_scores)
            else:
                loss = loss_func(pos_scores, None)
                loss = loss + cfg.params.stay_positive_reg_lambda * model.reg_score(pos_triplets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_epoch_losses.append(loss.item())

        if cfg.run_args.use_weights:
            for pairs in weighted_triples_loader:
                pairs = pairs.to(device).long()
                pos_scores = model.get_weighted_score(pairs)
                loss = loss_func(pos_scores)
                opt.zero_grad()
                loss.backward()
                opt.step()
        if cfg.run_args.weak_nodes_mse:
            for node in kg.weak_nodes:
                if (len(kg.nearest[node]) > 0):
                    x = torch.cat([torch.cat(
                        [model.ent_1(torch.tensor(node, device=device)).reshape(1, -1),
                         model.ent_2(torch.tensor(node, device=device)).reshape(1, -1),
                         model.ent_3(torch.tensor(node, device=device)).reshape(1, -1)], dim=1
                    ) for closest in kg.nearest[node]], dim=1)

                    # print(model.ent_1(torch.tensor(node, device=device)).shape())
                    # [for closest in kg.nearest[node]]

                    #     reg1 = model.regularization_const * torch.mean(torch.abs(h_3) ** 3, dim=1) +
                    #  torch.mean(torch.abs(t_1) ** 3, dim=1) +
                    #  torch.mean(torch.abs(t_2) ** 3, dim=1)

                    targets = torch.cat([torch.cat(
                        [model.ent_1(torch.tensor(closest, device=device)).reshape(1, -1),
                         model.ent_2(torch.tensor(closest, device=device)).reshape(1, -1),
                         model.ent_3(torch.tensor(closest, device=device)).reshape(1, -1)], dim=1
                    ) for closest in kg.nearest[node]], dim=1)
                    loss = mse_loss(x, targets)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        if not epoch % cfg.run_args.print_progress_every:
            switch_grad_mode(model, requires_grad=False)
            switch_model_mode(model, train=False)

            val_loss, val_metrics = evaluation(
                val_pos_loader, val_neg_loader, model, device, loss_func,
                metrics_separately=True, psi=args_psi)

            _, val_metrics_norm = evaluation(
                val_pos_loader, val_neg_loader, model, device, loss_func,
                metrics_separately=False, psi=args_psi)

            log_addition = "" if cv_iter is None else ("_" + str(cv_iter))

            val_roc = val_metrics['auc_roc']

            print(f"AUROC: {val_roc}, AUPRC: {val_metrics['auprc']}")

            if cfg.run_args.log:

                neptune.log_metric("train_loss" + log_addition,
                                   np.mean(train_epoch_losses),
                                   timestamp=time.time())
                neptune.log_metric("val_all_loss" + log_addition, val_loss,
                                   timestamp=time.time())

                for metric, value in val_metrics.items():
                    neptune.log_metric(f'val_all_{metric}' + log_addition,
                                       value,
                                       timestamp=time.time())

                neptune.log_metric(f'val_all_auc_roc_norm' + log_addition,
                                   val_metrics_norm['auc_roc'],
                                   timestamp=time.time())
            if cfg.run_args.val_regenerate:
                val_neg_data.regenerate()
        if cfg.run_args.save_every > 0:
            if not epoch % cfg.run_args.save_every:
                switch_grad_mode(model, requires_grad=False)
                switch_model_mode(model, train=False)
                path = model_save_path + '/' + str(epoch) + '.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                }, path)

        if cfg.run_args.early_stop:
            if val_roc - max_val_roc < cfg.params.early_stopping_epsilon:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
            if val_roc > max_val_roc:
                max_val_roc = val_roc
                best_model_params = deepcopy(model.state_dict())

            if early_stop_counter == cfg.params.early_stopping_k:
                print(f"Early stop at epoch {epoch - cfg.params.early_stopping_k}")
                max_epoch = epoch
                break
        else:
            if val_roc > max_val_roc:
                max_val_roc = val_roc
                best_model_params = deepcopy(model.state_dict())

    print('Test')

    # Load best model from training
    model.load_state_dict(best_model_params)

    path = model_save_path + '/' + 'best_params.pt'
    torch.save({
        'epoch': max_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, path)

    switch_grad_mode(model, requires_grad=False)
    switch_model_mode(model, train=False)

    # Loss and classic metrics
    test_loss, test_metrics = evaluation(
        test_pos_loader, test_neg_loader, model, device, loss_func,
        metrics_separately=True, psi=args_psi)
    _, test_metrics_norm = evaluation(
        test_pos_loader, test_neg_loader, model, device, loss_func,
        metrics_separately=False, psi=args_psi)
    if cfg.run_args.log:
        neptune.log_metric("test_all_loss" + log_addition, test_loss, timestamp=time.time())
        for metric in test_metrics.keys():
            neptune.log_metric(f'test_all_{metric}' + log_addition, test_metrics[metric],
                               timestamp=time.time())
        neptune.log_metric(f'test_all_auc_roc_norm' + log_addition,
                           test_metrics_norm['auc_roc'],
                           timestamp=time.time())

    # Ranking metrics
    test_rank = evaluation_ranking(
        np.array(kg.get_data_by_type('test')), kg.get_num_of_ent("test"),
        model, device, known_triples=kg.get_drugs_array(), verbose=1)['filtered']['avg']
    if cfg.run_args.log:
        for metric in test_rank.keys():
            neptune.log_metric(f'test_all_{metric}' + log_addition, test_rank[metric],
                               timestamp=time.time())


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    hydra.output_subdir = None
    if cfg.run_args.log:
        import neptune

        neptune.init('Pollutants/sandbox')
        neptune_experiment_name = cfg.run_args.experiment_name
        exp = neptune.create_experiment(name=neptune_experiment_name,
                                        params=cfg,
                                        upload_stdout=True,
                                        upload_stderr=True,
                                        send_hardware_metrics=True,
                                        upload_source_files='**/*.py')

        if cfg.run_args.gpu:
            neptune.append_tag('gpu')
        if cfg.run_args.use_proteins:
            neptune.append_tag('proteins')
        if cfg.params.mult_run_folds:
            neptune.append_tag('multiple_run')
        if cfg.run_args.use_weights:
            neptune.append_tag(f"weights_{cfg.run_args.use_weights}")
        if cfg.run_args.use_embeddings_init:
            neptune.append_tag(f"init_{cfg.run_args.use_embeddings_init}")
        if cfg.run_args.similarity:
            neptune.append_tag('similarity_' + cfg.run_args.similarity)
        if cfg.run_args.weak_nodes_mse:
            neptune.append_tag(f'weak_nodes_mse_{cfg.run_args.n_closest}')
        if cfg.run_args.weak_node_list:
            neptune.append_tag(f'weak_node_list_{cfg.run_args.weak_node_list}')
        neptune.append_tag('trivec')
    use_cuda = cfg.run_args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if cfg.run_args.gpu else "cpu")
    print(f'Use device: {device}')

    # Train
    if cfg.run_args.log:
        model_save_path = (cfg.data_const.work_dir + cfg.data_const.save_path +
                           '/' + exp.id)
    else:
        t = datetime.now()
        model_save_path = (cfg.data_const.work_dir + cfg.data_const.save_path +
                           '/' + cfg.run_args.experiment_name + " " + t.strftime("%d_%m_%Y") + t.strftime("_%H_%M_%S"))

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if cfg.run_args.use_weights:
        if cfg.run_args.similarity:
            similarity_type = cfg.run_args.similarity
        else:
            similarity_type = cfg['embedding_data'][cfg.run_args.use_weights]['similarity_score']
        if cfg.run_args.similarity == "parzen":
            if cfg.run_args.weak_nodes_mse:
                kg = MSEParzenKnowledgeGraph(cfg=cfg,
                                             data_path=cfg.data_const.work_dir,
                                             use_proteins=cfg.run_args.use_proteins,
                                             use_proteins_on_validation=False,
                                             use_reversed_edges=cfg.run_args.reversed,
                                             descriptors_type=cfg.run_args.use_weights,
                                             use_saves=cfg.run_args.use_saves,
                                             n_closest=cfg.run_args.n_closest,
                                             weak_node_list=cfg.run_args.weak_node_list)
            else:
                kg = ParzenSimilarityKnowledgeGraph(cfg=cfg,
                                                    data_path=cfg.data_const.work_dir,
                                                    use_proteins=cfg.run_args.use_proteins,
                                                    use_proteins_on_validation=False,
                                                    use_reversed_edges=cfg.run_args.reversed,
                                                    descriptors_type=cfg.run_args.use_weights)
        else:
            if cfg.run_args.weak_nodes_mse:
                kg = MSEDegreeSimilarityKnowledgeGraph(cfg=cfg,
                                                       data_path=cfg.data_const.work_dir,
                                                       use_proteins=cfg.run_args.use_proteins,
                                                       use_proteins_on_validation=False,
                                                       use_reversed_edges=cfg.run_args.reversed,
                                                       descriptors_type=cfg.run_args.use_weights,
                                                       similarity_type=similarity_type,
                                                       use_saves=cfg.run_args.use_saves,
                                                       n_closest=cfg.run_args.n_closest,
                                                       weak_node_list=cfg.run_args.weak_node_list)
            else:
                kg = DegreeSimilarityKnowledgeGraph(cfg=cfg,
                                                    data_path=cfg.data_const.work_dir,
                                                    use_proteins=cfg.run_args.use_proteins,
                                                    use_proteins_on_validation=False,
                                                    use_reversed_edges=cfg.run_args.reversed,
                                                    descriptors_type=cfg.run_args.use_weights,
                                                    similarity_type=similarity_type)
    elif cfg.run_args.weak_nodes_mse:
        kg = MSEKnowledgeGraph(cfg=cfg,
                               data_path=cfg.data_const.work_dir,
                               use_proteins=cfg.run_args.use_proteins,
                               use_proteins_on_validation=False,
                               use_reversed_edges=cfg.run_args.reversed, use_saves=cfg.run_args.use_saves,
                               n_closest=cfg.run_args.n_closest, weak_node_list=cfg.run_args.weak_node_list)
    else:
        kg = KnowledgeGraph(cfg=cfg,
                            data_path=cfg.data_const.work_dir,
                            use_proteins=cfg.run_args.use_proteins,
                            use_proteins_on_validation=False,
                            use_reversed_edges=cfg.run_args.reversed)

    if cfg.params.mult_run_folds > 0:
        for run_iter in range(0, cfg.params.mult_run_folds):
            print(f"Iter: {run_iter}/{cfg.params.mult_run_folds}")
            run_train_iteration(kg, model_save_path, cfg, device, run_iter)
    else:
        run_train_iteration(kg, model_save_path, cfg, device)

    if cfg.run_args.log:
        neptune.stop()


if __name__ == '__main__':
    main()