"""
Sistema de Trading Quantitativo - SHAP Analyzer
Versão: 2.01
Descrição: Análise de importância de features usando SHAP
"""

import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import boto3
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """
    Classe responsável pela análise de importância de features usando SHAP.
    Fornece insights sobre quais features são mais importantes para as predições do modelo.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o SHAPAnalyzer.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config.get('data_bucket_name')
        
        # Configurações do SHAP
        self.shap_config = config.get('shap_config', {
            'max_samples_for_analysis': 1000,
            'background_samples': 100,
            'plot_top_features': 20,
            'save_plots': True
        })
        
    def analyze_feature_importance(self, model: lgb.LGBMClassifier,
                                 X_train: pd.DataFrame,
                                 X_val: pd.DataFrame,
                                 feature_names: List[str]) -> Dict:
        """
        Analisa importância das features usando SHAP.
        
        Args:
            model: Modelo LightGBM treinado
            X_train: Features de treinamento
            X_val: Features de validação
            feature_names: Nomes das features
            
        Returns:
            Dicionário com análise SHAP completa
        """
        logger.info("Iniciando análise SHAP de importância de features")
        
        try:
            # Prepara dados para análise
            analysis_data = self._prepare_analysis_data(X_train, X_val)
            background_data = self._prepare_background_data(X_train)
            
            # Cria explainer SHAP
            explainer = self._create_shap_explainer(model, background_data)
            
            # Calcula valores SHAP
            shap_values = self._calculate_shap_values(explainer, analysis_data)
            
            # Analisa importância global
            global_importance = self._analyze_global_importance(shap_values, feature_names)
            
            # Analisa importância por classe
            class_importance = self._analyze_class_importance(shap_values, feature_names)
            
            # Analisa interações entre features
            interaction_analysis = self._analyze_feature_interactions(
                explainer, analysis_data, feature_names
            )
            
            # Analisa dependências
            dependency_analysis = self._analyze_feature_dependencies(
                shap_values, analysis_data, feature_names
            )
            
            # Gera insights
            insights = self._generate_feature_insights(
                global_importance, class_importance, interaction_analysis
            )
            
            # Gera plots se configurado
            plot_paths = []
            if self.shap_config['save_plots']:
                plot_paths = self._generate_shap_plots(
                    shap_values, analysis_data, feature_names, explainer
                )
            
            analysis_result = {
                'analysis_timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'total_features': len(feature_names),
                'samples_analyzed': len(analysis_data),
                'global_importance': global_importance,
                'class_importance': class_importance,
                'interaction_analysis': interaction_analysis,
                'dependency_analysis': dependency_analysis,
                'insights': insights,
                'plot_paths': plot_paths
            }
            
            logger.info("Análise SHAP concluída")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erro na análise SHAP: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _prepare_analysis_data(self, X_train: pd.DataFrame, 
                             X_val: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados para análise SHAP.
        
        Args:
            X_train: Features de treinamento
            X_val: Features de validação
            
        Returns:
            DataFrame preparado para análise
        """
        # Combina dados de treino e validação
        combined_data = pd.concat([X_train, X_val], ignore_index=True)
        
        # Limita número de amostras se necessário
        max_samples = self.shap_config['max_samples_for_analysis']
        if len(combined_data) > max_samples:
            # Amostragem estratificada (mantém distribuição temporal)
            step = len(combined_data) // max_samples
            analysis_data = combined_data.iloc[::step].copy()
        else:
            analysis_data = combined_data.copy()
        
        logger.debug(f"Dados preparados para análise: {len(analysis_data)} amostras")
        return analysis_data
    
    def _prepare_background_data(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados de background para SHAP.
        
        Args:
            X_train: Features de treinamento
            
        Returns:
            DataFrame com dados de background
        """
        background_samples = self.shap_config['background_samples']
        
        if len(X_train) > background_samples:
            # Amostragem aleatória para background
            background_data = X_train.sample(n=background_samples, random_state=42)
        else:
            background_data = X_train.copy()
        
        logger.debug(f"Dados de background preparados: {len(background_data)} amostras")
        return background_data
    
    def _create_shap_explainer(self, model: lgb.LGBMClassifier,
                             background_data: pd.DataFrame) -> shap.Explainer:
        """
        Cria explainer SHAP apropriado para o modelo.
        
        Args:
            model: Modelo treinado
            background_data: Dados de background
            
        Returns:
            Explainer SHAP
        """
        logger.debug("Criando explainer SHAP")
        
        try:
            # Para LightGBM, usa TreeExplainer
            explainer = shap.TreeExplainer(model._Booster)
            logger.debug("TreeExplainer criado com sucesso")
            return explainer
            
        except Exception as e:
            logger.warning(f"Erro ao criar TreeExplainer: {str(e)}")
            
            # Fallback para Explainer genérico
            try:
                def model_predict(X):
                    return model._Booster.predict(X, num_iteration=model._Booster.best_iteration)
                
                explainer = shap.Explainer(model_predict, background_data)
                logger.debug("Explainer genérico criado como fallback")
                return explainer
                
            except Exception as e2:
                logger.error(f"Erro ao criar explainer genérico: {str(e2)}")
                raise
    
    def _calculate_shap_values(self, explainer: shap.Explainer,
                             analysis_data: pd.DataFrame) -> np.ndarray:
        """
        Calcula valores SHAP.
        
        Args:
            explainer: Explainer SHAP
            analysis_data: Dados para análise
            
        Returns:
            Array com valores SHAP
        """
        logger.debug("Calculando valores SHAP")
        
        try:
            shap_values = explainer.shap_values(analysis_data)
            
            # Para modelos multiclasse, shap_values é uma lista
            if isinstance(shap_values, list):
                logger.debug(f"Valores SHAP calculados para {len(shap_values)} classes")
            else:
                logger.debug("Valores SHAP calculados")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Erro ao calcular valores SHAP: {str(e)}")
            raise
    
    def _analyze_global_importance(self, shap_values: np.ndarray,
                                 feature_names: List[str]) -> Dict:
        """
        Analisa importância global das features.
        
        Args:
            shap_values: Valores SHAP
            feature_names: Nomes das features
            
        Returns:
            Dicionário com importância global
        """
        logger.debug("Analisando importância global")
        
        if isinstance(shap_values, list):
            # Para modelos multiclasse, calcula importância média
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Cria ranking de importância
        feature_importance = dict(zip(feature_names, mean_abs_shap))
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        # Calcula estatísticas
        total_importance = sum(mean_abs_shap)
        relative_importance = {
            feature: importance / total_importance 
            for feature, importance in sorted_importance.items()
        }
        
        # Top features
        top_features = list(sorted_importance.keys())[:self.shap_config['plot_top_features']]
        
        return {
            'feature_importance': sorted_importance,
            'relative_importance': relative_importance,
            'top_features': top_features,
            'total_importance': total_importance,
            'feature_count': len(feature_names)
        }
    
    def _analyze_class_importance(self, shap_values: np.ndarray,
                                feature_names: List[str]) -> Dict:
        """
        Analisa importância por classe.
        
        Args:
            shap_values: Valores SHAP
            feature_names: Nomes das features
            
        Returns:
            Dicionário com importância por classe
        """
        logger.debug("Analisando importância por classe")
        
        if not isinstance(shap_values, list):
            return {'note': 'Class-specific analysis not available for binary classification'}
        
        class_names = ['Sell', 'Neutral', 'Buy']
        class_importance = {}
        
        for i, class_name in enumerate(class_names):
            if i < len(shap_values):
                class_shap = np.abs(shap_values[i]).mean(axis=0)
                class_feature_importance = dict(zip(feature_names, class_shap))
                sorted_class_importance = dict(sorted(class_feature_importance.items(),
                                                    key=lambda x: x[1], reverse=True))
                
                class_importance[class_name] = {
                    'feature_importance': sorted_class_importance,
                    'top_features': list(sorted_class_importance.keys())[:10],
                    'total_importance': sum(class_shap)
                }
        
        # Analisa features mais discriminativas entre classes
        discriminative_features = self._find_discriminative_features(shap_values, feature_names)
        
        return {
            'class_specific_importance': class_importance,
            'discriminative_features': discriminative_features
        }
    
    def _find_discriminative_features(self, shap_values: List[np.ndarray],
                                    feature_names: List[str]) -> Dict:
        """
        Encontra features mais discriminativas entre classes.
        
        Args:
            shap_values: Lista de valores SHAP por classe
            feature_names: Nomes das features
            
        Returns:
            Dicionário com features discriminativas
        """
        if len(shap_values) < 2:
            return {}
        
        # Calcula variância da importância entre classes para cada feature
        importance_by_class = []
        for sv in shap_values:
            importance_by_class.append(np.abs(sv).mean(axis=0))
        
        importance_matrix = np.array(importance_by_class)
        feature_variance = np.var(importance_matrix, axis=0)
        
        # Features com maior variância são mais discriminativas
        discriminative_scores = dict(zip(feature_names, feature_variance))
        sorted_discriminative = dict(sorted(discriminative_scores.items(),
                                          key=lambda x: x[1], reverse=True))
        
        return {
            'discriminative_scores': sorted_discriminative,
            'top_discriminative_features': list(sorted_discriminative.keys())[:10]
        }
    
    def _analyze_feature_interactions(self, explainer: shap.Explainer,
                                    analysis_data: pd.DataFrame,
                                    feature_names: List[str]) -> Dict:
        """
        Analisa interações entre features.
        
        Args:
            explainer: Explainer SHAP
            analysis_data: Dados para análise
            feature_names: Nomes das features
            
        Returns:
            Dicionário com análise de interações
        """
        logger.debug("Analisando interações entre features")
        
        try:
            # Limita análise a top features para performance
            top_features_count = min(10, len(feature_names))
            
            # Calcula valores SHAP de interação (computacionalmente caro)
            # Usa amostra menor para interações
            sample_size = min(100, len(analysis_data))
            sample_data = analysis_data.sample(n=sample_size, random_state=42)
            
            # Para TreeExplainer, pode calcular interações
            if hasattr(explainer, 'shap_interaction_values'):
                interaction_values = explainer.shap_interaction_values(sample_data)
                
                # Analisa interações mais fortes
                strong_interactions = self._find_strong_interactions(
                    interaction_values, feature_names, top_features_count
                )
                
                return {
                    'interaction_analysis_available': True,
                    'strong_interactions': strong_interactions,
                    'sample_size_used': sample_size
                }
            else:
                return {
                    'interaction_analysis_available': False,
                    'note': 'Interaction analysis not supported for this explainer type'
                }
                
        except Exception as e:
            logger.warning(f"Erro na análise de interações: {str(e)}")
            return {
                'interaction_analysis_available': False,
                'error': str(e)
            }
    
    def _find_strong_interactions(self, interaction_values: np.ndarray,
                                feature_names: List[str],
                                top_count: int) -> List[Dict]:
        """
        Encontra interações mais fortes entre features.
        
        Args:
            interaction_values: Valores de interação SHAP
            feature_names: Nomes das features
            top_count: Número de top features a considerar
            
        Returns:
            Lista com interações mais fortes
        """
        if isinstance(interaction_values, list):
            # Para multiclasse, usa primeira classe
            interaction_values = interaction_values[0]
        
        # Calcula força média das interações
        mean_interactions = np.abs(interaction_values).mean(axis=0)
        
        strong_interactions = []
        
        # Analisa interações entre top features
        for i in range(min(top_count, len(feature_names))):
            for j in range(i + 1, min(top_count, len(feature_names))):
                interaction_strength = mean_interactions[i, j]
                
                if interaction_strength > 0:  # Threshold mínimo
                    strong_interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': float(interaction_strength)
                    })
        
        # Ordena por força da interação
        strong_interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        return strong_interactions[:10]  # Top 10 interações
    
    def _analyze_feature_dependencies(self, shap_values: np.ndarray,
                                    analysis_data: pd.DataFrame,
                                    feature_names: List[str]) -> Dict:
        """
        Analisa dependências das features.
        
        Args:
            shap_values: Valores SHAP
            analysis_data: Dados de análise
            feature_names: Nomes das features
            
        Returns:
            Dicionário com análise de dependências
        """
        logger.debug("Analisando dependências das features")
        
        if isinstance(shap_values, list):
            # Para multiclasse, usa valores médios
            mean_shap_values = np.mean(shap_values, axis=0)
        else:
            mean_shap_values = shap_values
        
        dependencies = {}
        
        # Analisa top features
        top_features_count = min(10, len(feature_names))
        
        for i in range(top_features_count):
            feature_name = feature_names[i]
            feature_values = analysis_data.iloc[:, i].values
            feature_shap = mean_shap_values[:, i]
            
            # Calcula correlação entre valor da feature e seu SHAP value
            correlation = np.corrcoef(feature_values, feature_shap)[0, 1]
            
            # Analisa distribuição dos valores SHAP
            shap_stats = {
                'mean': float(np.mean(feature_shap)),
                'std': float(np.std(feature_shap)),
                'min': float(np.min(feature_shap)),
                'max': float(np.max(feature_shap))
            }
            
            dependencies[feature_name] = {
                'value_shap_correlation': float(correlation) if not np.isnan(correlation) else 0,
                'shap_statistics': shap_stats,
                'feature_statistics': {
                    'mean': float(np.mean(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values))
                }
            }
        
        return dependencies
    
    def _generate_feature_insights(self, global_importance: Dict,
                                 class_importance: Dict,
                                 interaction_analysis: Dict) -> Dict:
        """
        Gera insights sobre as features.
        
        Args:
            global_importance: Importância global
            class_importance: Importância por classe
            interaction_analysis: Análise de interações
            
        Returns:
            Dicionário com insights
        """
        logger.debug("Gerando insights das features")
        
        insights = {
            'key_findings': [],
            'recommendations': [],
            'feature_categories': {}
        }
        
        # Analisa concentração de importância
        top_5_importance = sum(list(global_importance['relative_importance'].values())[:5])
        if top_5_importance > 0.7:
            insights['key_findings'].append(
                f"Top 5 features representam {top_5_importance:.1%} da importância total - modelo pode estar overfitting"
            )
            insights['recommendations'].append("Considerar regularização adicional ou feature selection")
        
        # Analisa features mais importantes
        top_feature = list(global_importance['feature_importance'].keys())[0]
        top_importance = list(global_importance['relative_importance'].values())[0]
        
        insights['key_findings'].append(
            f"Feature mais importante: {top_feature} ({top_importance:.1%} da importância total)"
        )
        
        # Categoriza features por tipo
        feature_categories = self._categorize_features(global_importance['top_features'])
        insights['feature_categories'] = feature_categories
        
        # Analisa balanceamento entre categorias
        category_balance = self._analyze_category_balance(feature_categories)
        insights['category_balance'] = category_balance
        
        # Recomendações baseadas nas categorias
        if category_balance.get('ichimoku_ratio', 0) > 0.5:
            insights['recommendations'].append("Modelo fortemente dependente de Ichimoku - considerar diversificar indicadores")
        
        if category_balance.get('volume_ratio', 0) < 0.1:
            insights['recommendations'].append("Baixa importância de features de volume - verificar qualidade dos dados de volume")
        
        return insights
    
    def _categorize_features(self, feature_names: List[str]) -> Dict:
        """
        Categoriza features por tipo de indicador.
        
        Args:
            feature_names: Lista de nomes das features
            
        Returns:
            Dicionário com categorização
        """
        categories = {
            'ichimoku': [],
            'rsi': [],
            'atr': [],
            'candlestick': [],
            'volume': [],
            'other': []
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['tenkan', 'kijun', 'senkou', 'chikou', 'cloud', 'kumo']):
                categories['ichimoku'].append(feature)
            elif 'rsi' in feature_lower:
                categories['rsi'].append(feature)
            elif 'atr' in feature_lower:
                categories['atr'].append(feature)
            elif any(keyword in feature_lower for keyword in ['body', 'shadow', 'hammer', 'doji', 'engulfing', 'star', 'gap']):
                categories['candlestick'].append(feature)
            elif any(keyword in feature_lower for keyword in ['volume', 'obv', 'vpt', 'ad_line']):
                categories['volume'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def _analyze_category_balance(self, feature_categories: Dict) -> Dict:
        """
        Analisa balanceamento entre categorias de features.
        
        Args:
            feature_categories: Categorias de features
            
        Returns:
            Dicionário com análise de balanceamento
        """
        total_features = sum(len(features) for features in feature_categories.values())
        
        if total_features == 0:
            return {}
        
        balance = {}
        for category, features in feature_categories.items():
            balance[f'{category}_count'] = len(features)
            balance[f'{category}_ratio'] = len(features) / total_features
        
        return balance
    
    def _generate_shap_plots(self, shap_values: np.ndarray,
                           analysis_data: pd.DataFrame,
                           feature_names: List[str],
                           explainer: shap.Explainer) -> List[str]:
        """
        Gera plots SHAP.
        
        Args:
            shap_values: Valores SHAP
            analysis_data: Dados de análise
            feature_names: Nomes das features
            explainer: Explainer SHAP
            
        Returns:
            Lista de paths dos plots gerados
        """
        logger.debug("Gerando plots SHAP")
        
        plot_paths = []
        
        try:
            # Plot 1: Summary plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                # Para multiclasse, plota primeira classe
                shap.summary_plot(shap_values[0], analysis_data, 
                                feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, analysis_data, 
                                feature_names=feature_names, show=False)
            
            summary_path = '/tmp/shap_summary_plot.png'
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(summary_path)
            
            # Plot 2: Feature importance
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Ordena features por importância
            feature_importance = dict(zip(feature_names, mean_shap))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Plota top features
            top_n = min(20, len(sorted_features))
            top_features = sorted_features[:top_n]
            
            features, importances = zip(*top_features)
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            importance_path = '/tmp/shap_feature_importance.png'
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(importance_path)
            
            # Plot 3: Dependence plot para top feature
            if len(feature_names) > 0:
                plt.figure(figsize=(10, 6))
                top_feature_idx = np.argmax(mean_shap)
                
                if isinstance(shap_values, list):
                    shap.dependence_plot(top_feature_idx, shap_values[0], analysis_data,
                                       feature_names=feature_names, show=False)
                else:
                    shap.dependence_plot(top_feature_idx, shap_values, analysis_data,
                                       feature_names=feature_names, show=False)
                
                dependence_path = '/tmp/shap_dependence_plot.png'
                plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(dependence_path)
            
            logger.debug(f"Gerados {len(plot_paths)} plots SHAP")
            
        except Exception as e:
            logger.warning(f"Erro ao gerar plots SHAP: {str(e)}")
        
        return plot_paths
    
    def save_shap_analysis(self, analysis_result: Dict,
                          symbol: str, timeframe: str) -> str:
        """
        Salva análise SHAP no S3.
        
        Args:
            analysis_result: Resultado da análise SHAP
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Path do arquivo salvo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shap_analysis_{symbol}_{timeframe}_{timestamp}.json"
        s3_key = f"shap_analysis/{filename}"
        
        try:
            # Converte para JSON
            analysis_json = json.dumps(analysis_result, indent=2, default=str)
            
            # Upload para S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=analysis_json,
                ContentType='application/json'
            )
            
            s3_path = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Análise SHAP salva em: {s3_path}")
            return s3_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar análise SHAP: {str(e)}")
            raise
    
    def upload_plots_to_s3(self, plot_paths: List[str],
                          symbol: str, timeframe: str) -> List[str]:
        """
        Faz upload dos plots para S3.
        
        Args:
            plot_paths: Lista de paths locais dos plots
            symbol: Par de trading
            timeframe: Timeframe
            
        Returns:
            Lista de paths S3 dos plots
        """
        s3_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, local_path in enumerate(plot_paths):
            if os.path.exists(local_path):
                try:
                    # Nome do arquivo no S3
                    plot_name = f"shap_plot_{symbol}_{timeframe}_{timestamp}_{i}.png"
                    s3_key = f"shap_plots/{plot_name}"
                    
                    # Upload para S3
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                    
                    s3_path = f"s3://{self.bucket_name}/{s3_key}"
                    s3_paths.append(s3_path)
                    
                    # Remove arquivo local
                    os.remove(local_path)
                    
                except Exception as e:
                    logger.warning(f"Erro ao fazer upload do plot {local_path}: {str(e)}")
        
        logger.info(f"Upload de {len(s3_paths)} plots concluído")
        return s3_paths

