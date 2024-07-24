import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, screen, waitFor } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { getExperimentApi, getRunApi, updateRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';
import { merge } from 'lodash';
import { ReduxState } from '../../../redux-types';
import { DeepPartial } from 'redux';
import { RunPage } from './RunPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { RunInfoEntity } from '../../types';
import userEvent from '@testing-library/user-event-14';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';

const mockAction = (id: string) => ({ type: 'action', payload: Promise.resolve(), meta: { id } });

jest.mock('../../actions', () => ({
  getExperimentApi: jest.fn(() => mockAction('experiment_request')),
  getRunApi: jest.fn(() => mockAction('run_request')),
  updateRunApi: jest.fn(() => mockAction('run_request')),
}));

jest.mock('../../../model-registry/actions', () => ({
  searchModelVersionsApi: jest.fn(() => mockAction('models_request')),
}));

const testRunUuid = 'test-run-uuid';
const testExperimentId = '12345';
const testRunInfo: Partial<RunInfoEntity> = {
  runName: 'Test run Name',
  experimentId: testExperimentId,
};

describe('RunPage', () => {
  const mountComponent = (
    entities: DeepPartial<ReduxState['entities']> = {},
    apis: DeepPartial<ReduxState['apis']> = {},
  ) => {
    const state: DeepPartial<ReduxState> = {
      entities: merge(
        {
          artifactRootUriByRunUuid: {},
          runInfosByUuid: {},
          experimentsById: {},
          tagsByRunUuid: {},
          latestMetricsByRunUuid: {},
          runDatasetsByUuid: {},
          paramsByRunUuid: {},
          modelVersionsByRunUuid: {},
        },
        entities,
      ),
      apis: merge(
        {
          experiment_request: { active: true },
          run_request: { active: true },
        },
        apis,
      ),
    };

    const renderResult = renderWithIntl(
      <MockedReduxStoreProvider state={state}>
        <TestRouter
          initialEntries={[`/experiment/${testExperimentId}/run/${testRunUuid}`]}
          routes={[testRoute(<RunPage />, '/experiment/:experimentId/run/:runUuid')]}
        />
      </MockedReduxStoreProvider>,
    );

    return renderResult;
  };

  beforeEach(() => {
    jest.mocked(getRunApi).mockClear();
    jest.mocked(getExperimentApi).mockClear();
    jest.mocked(searchModelVersionsApi).mockClear();
    jest.mocked(updateRunApi).mockClear();
  });

  test('Start fetching run when store is empty and experiment and indicate loading state', async () => {
    mountComponent();

    await waitFor(() => {
      expect(screen.getByText('Run page loading')).toBeInTheDocument();
    });

    expect(getRunApi).toBeCalledWith(testRunUuid);
    expect(getExperimentApi).toBeCalledWith(testExperimentId);
    expect(searchModelVersionsApi).toBeCalledWith({ run_id: testRunUuid });
  });

  const entitiesWithMockRun = {
    runInfosByUuid: { [testRunUuid]: testRunInfo },
    experimentsById: {
      [testExperimentId]: { experimentId: testExperimentId, name: 'Test experiment name' },
    },
    tagsByRunUuid: { [testRunUuid]: {} },
    latestMetricsByRunUuid: {},
    runDatasetsByUuid: {},
    paramsByRunUuid: {},
    modelVersionsByRunUuid: {},
  };

  test('Do not display loading state when run and experiments are already loaded', async () => {
    mountComponent(entitiesWithMockRun);

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Test run Name/ })).toBeInTheDocument();
    });

    expect(getRunApi).not.toBeCalled();
    expect(getExperimentApi).not.toBeCalled();
    expect(searchModelVersionsApi).toBeCalled();
  });

  test('Attempt to rename the run', async () => {
    mountComponent(entitiesWithMockRun);

    await waitFor(() => {
      expect(screen.getByLabelText('Open header dropdown menu')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('Open header dropdown menu'));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));
    await userEvent.clear(screen.getByTestId('rename-modal-input'));
    await userEvent.type(screen.getByTestId('rename-modal-input'), 'brand_new_run_name');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(updateRunApi).toBeCalledWith('test-run-uuid', 'brand_new_run_name', expect.anything());
  });

  test('Display 404 page in case of missing run', async () => {
    const runFetchError = new ErrorWrapper({ error_code: 'RESOURCE_DOES_NOT_EXIST' });

    mountComponent({}, { run_request: { active: false, error: runFetchError } });

    await waitFor(() => {
      expect(screen.getByText(/Run ID test-run-uuid does not exist/)).toBeInTheDocument();
    });
  });
});
