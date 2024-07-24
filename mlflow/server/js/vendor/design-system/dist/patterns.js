import _isUndefined from 'lodash/isUndefined';
import { useMemo, useState, useCallback } from 'react';
import { css, createElement } from '@emotion/react';
import _pick from 'lodash/pick';
import { jsxs, Fragment, jsx } from '@emotion/react/jsx-runtime';
import { S as Stepper } from './Stepper-431a4a97.js';
import { a as useDesignSystemTheme, w as getShadowScrollStyles, B as Button, h as addDebugOutlineIfEnabled, k as Root, T as Trigger, l as Content } from './Typography-c0049677.js';
import _compact from 'lodash/compact';
import { S as Spacer, a as useMediaQuery, L as ListIcon, M as Modal } from './useMediaQuery-3b982e75.js';
import 'antd';
import '@radix-ui/react-popover';
import '@radix-ui/react-tooltip';
import '@ant-design/icons';
import 'lodash/isNil';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';
import 'lodash/memoize';
import '@emotion/unitless';

function useStepperStepsFromWizardSteps(wizardSteps, currentStepIdx) {
  return useMemo(() => wizardSteps.map((wizardStep, stepIdx) => ({
    ..._pick(wizardStep, ['title', 'description', 'status']),
    additionalVerticalContent: wizardStep.additionalHorizontalLayoutStepContent,
    clickEnabled: _isUndefined(wizardStep.clickEnabled) ? isWizardStepEnabled(wizardSteps, stepIdx, currentStepIdx, wizardStep.status) : wizardStep.clickEnabled
  })), [currentStepIdx, wizardSteps]);
}
function isWizardStepEnabled(steps, stepIdx, currentStepIdx, status) {
  if (stepIdx < currentStepIdx || status === 'completed' || status === 'error' || status === 'warning') {
    return true;
  }

  // if every step before stepIdx is completed then the step is enabled
  return steps.slice(0, stepIdx - 1).every(step => step.status === 'completed');
}

function HorizontalWizardStepsContent(_ref) {
  var _wizardSteps$currentS, _wizardSteps$currentS2;
  let {
    steps: wizardSteps,
    currentStepIndex,
    localizeStepNumber,
    enableClickingToSteps,
    goToStep
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex);
  const expandContentToFullHeight = (_wizardSteps$currentS = wizardSteps[currentStepIndex].expandContentToFullHeight) !== null && _wizardSteps$currentS !== void 0 ? _wizardSteps$currentS : true;
  const disableDefaultScrollBehavior = (_wizardSteps$currentS2 = wizardSteps[currentStepIndex].disableDefaultScrollBehavior) !== null && _wizardSteps$currentS2 !== void 0 ? _wizardSteps$currentS2 : false;
  return jsxs(Fragment, {
    children: [jsx(Stepper, {
      currentStepIndex: currentStepIndex,
      direction: "horizontal",
      localizeStepNumber: localizeStepNumber,
      steps: stepperSteps,
      responsive: false,
      onStepClicked: enableClickingToSteps ? goToStep : undefined
    }), jsx("div", {
      css: /*#__PURE__*/css({
        marginTop: theme.spacing.md,
        flexGrow: expandContentToFullHeight ? 1 : undefined,
        overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
        ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {})
      }, process.env.NODE_ENV === "production" ? "" : ";label:HorizontalWizardStepsContent;"),
      children: wizardSteps[currentStepIndex].content
    })]
  });
}

/* eslint-disable react/no-unused-prop-types */
// eslint doesn't like passing props as object through to a function
// disabling to avoid a bunch of duplicate code.
function _EMOTION_STRINGIFIED_CSS_ERROR__() { return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop)."; }
var _ref2 = process.env.NODE_ENV === "production" ? {
  name: "1ff36h2",
  styles: "flex-grow:1"
} : {
  name: "1vyl5dv-getWizardFooterButtons",
  styles: "flex-grow:1;label:getWizardFooterButtons;",
  toString: _EMOTION_STRINGIFIED_CSS_ERROR__
};
// Buttons are returned in order with primary button last
function getWizardFooterButtons(_ref) {
  var _extraFooterButtonsRi;
  let {
    goToNextStepOrDone,
    isLastStep,
    currentStepIndex,
    goToPreviousStep,
    busyValidatingNextStep,
    nextButtonDisabled,
    nextButtonLoading,
    nextButtonContentOverride,
    previousButtonContentOverride,
    previousStepButtonHidden,
    previousButtonDisabled,
    previousButtonLoading,
    cancelButtonContent,
    cancelStepButtonHidden,
    nextButtonContent,
    previousButtonContent,
    doneButtonContent,
    extraFooterButtonsLeft,
    extraFooterButtonsRight,
    onCancel,
    moveCancelToOtherSide
  } = _ref;
  return _compact([!cancelStepButtonHidden && (moveCancelToOtherSide ? jsx("div", {
    css: _ref2,
    children: jsx(CancelButton, {
      onCancel: onCancel,
      cancelButtonContent: cancelButtonContent
    })
  }, "cancel") : jsx(CancelButton, {
    onCancel: onCancel,
    cancelButtonContent: cancelButtonContent
  }, "cancel")), currentStepIndex > 0 && !previousStepButtonHidden && jsx(Button, {
    onClick: goToPreviousStep,
    type: "tertiary",
    disabled: previousButtonDisabled,
    loading: previousButtonLoading,
    componentId: "codegen_dubois_src_wizard_wizardfooter_previous",
    children: previousButtonContentOverride ? previousButtonContentOverride : previousButtonContent
  }, "previous"), extraFooterButtonsLeft && extraFooterButtonsLeft.map((buttonProps, index) => createElement(Button, {
    ...buttonProps,
    type: undefined,
    key: `extra-left-${index}`
  })), jsx(Button, {
    onClick: goToNextStepOrDone,
    disabled: nextButtonDisabled,
    loading: nextButtonLoading || busyValidatingNextStep,
    type: ((_extraFooterButtonsRi = extraFooterButtonsRight === null || extraFooterButtonsRight === void 0 ? void 0 : extraFooterButtonsRight.length) !== null && _extraFooterButtonsRi !== void 0 ? _extraFooterButtonsRi : 0) > 0 ? undefined : 'primary',
    componentId: "codegen_dubois_src_wizard_wizardfooter_next",
    children: nextButtonContentOverride ? nextButtonContentOverride : isLastStep ? doneButtonContent : nextButtonContent
  }, "next"), extraFooterButtonsRight && extraFooterButtonsRight.map((buttonProps, index) => createElement(Button, {
    ...buttonProps,
    type: index === extraFooterButtonsRight.length - 1 ? 'primary' : undefined,
    key: `extra-right-${index}`
  }))]);
}
function CancelButton(_ref3) {
  let {
    onCancel,
    cancelButtonContent
  } = _ref3;
  return jsx(Button, {
    onClick: onCancel,
    type: "tertiary",
    componentId: "codegen_dubois_src_wizard_wizardfooter_cancel",
    children: cancelButtonContent
  }, "cancel");
}

function HorizontalWizardContent(_ref) {
  let {
    width,
    height,
    steps,
    currentStepIndex,
    localizeStepNumber,
    onStepsChange,
    enableClickingToSteps,
    ...footerProps
  } = _ref;
  return jsxs("div", {
    css: /*#__PURE__*/css({
      width,
      height,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'flex-start'
    }, process.env.NODE_ENV === "production" ? "" : ";label:HorizontalWizardContent;"),
    ...addDebugOutlineIfEnabled(),
    children: [jsx(HorizontalWizardStepsContent, {
      steps: steps,
      currentStepIndex: currentStepIndex,
      localizeStepNumber: localizeStepNumber,
      enableClickingToSteps: Boolean(enableClickingToSteps),
      goToStep: footerProps.goToStep
    }), jsx(Spacer, {
      size: "lg"
    }), jsx(WizardFooter, {
      currentStepIndex: currentStepIndex,
      ...steps[currentStepIndex],
      ...footerProps,
      moveCancelToOtherSide: true
    })]
  });
}
function WizardFooter(props) {
  const {
    theme
  } = useDesignSystemTheme();
  return jsx("div", {
    css: /*#__PURE__*/css({
      display: 'flex',
      flexDirection: 'row',
      justifyContent: 'flex-end',
      columnGap: theme.spacing.sm,
      paddingTop: theme.spacing.md,
      paddingBottom: theme.spacing.md,
      borderTop: `1px solid ${theme.colors.border}`
    }, process.env.NODE_ENV === "production" ? "" : ";label:WizardFooter;"),
    children: getWizardFooterButtons(props)
  });
}

const SmallFixedVerticalStepperWidth = 240;
const FixedVerticalStepperWidth = 280;
const MaxWizardContentWidth = 920;
const ExtraCompactButtonHeight = 32 + 24; // button height + gap

function VerticalWizardContent(_ref) {
  var _wizardSteps$currentS, _wizardSteps$currentS2;
  let {
    width,
    height,
    steps: wizardSteps,
    currentStepIndex,
    localizeStepNumber,
    onStepsChange,
    title,
    padding,
    verticalCompactButtonContent,
    enableClickingToSteps,
    ...footerProps
  } = _ref;
  const {
    theme
  } = useDesignSystemTheme();
  const stepperSteps = useStepperStepsFromWizardSteps(wizardSteps, currentStepIndex);
  const expandContentToFullHeight = (_wizardSteps$currentS = wizardSteps[currentStepIndex].expandContentToFullHeight) !== null && _wizardSteps$currentS !== void 0 ? _wizardSteps$currentS : true;
  const disableDefaultScrollBehavior = (_wizardSteps$currentS2 = wizardSteps[currentStepIndex].disableDefaultScrollBehavior) !== null && _wizardSteps$currentS2 !== void 0 ? _wizardSteps$currentS2 : false;
  const isCompact = useMediaQuery({
    query: `not (min-width: ${theme.responsive.breakpoints.lg}px)`
  }) && Boolean(verticalCompactButtonContent);
  return jsxs("div", {
    css: /*#__PURE__*/css({
      width,
      height: expandContentToFullHeight ? height : 'fit-content',
      maxHeight: '100%',
      display: 'flex',
      flexDirection: isCompact ? 'column' : 'row',
      gap: theme.spacing.lg,
      justifyContent: 'center'
    }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
    ...addDebugOutlineIfEnabled(),
    children: [!isCompact && jsxs("div", {
      css: /*#__PURE__*/css({
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        rowGap: theme.spacing.lg,
        paddingTop: theme.spacing.lg,
        paddingBottom: theme.spacing.lg,
        height: 'fit-content',
        width: SmallFixedVerticalStepperWidth,
        [`@media (min-width: ${theme.responsive.breakpoints.xl}px)`]: {
          width: FixedVerticalStepperWidth
        },
        overflowX: 'hidden'
      }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
      children: [title, jsx(Stepper, {
        currentStepIndex: currentStepIndex,
        direction: "vertical",
        localizeStepNumber: localizeStepNumber,
        steps: stepperSteps,
        responsive: false,
        onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
      })]
    }), isCompact && jsxs(Root, {
      children: [jsx(Trigger, {
        asChild: true,
        children: jsx("div", {
          children: jsx(Button, {
            icon: jsx(ListIcon, {}),
            componentId: "compact-vertical-wizard-show-stepper-popover",
            children: verticalCompactButtonContent === null || verticalCompactButtonContent === void 0 ? void 0 : verticalCompactButtonContent(currentStepIndex, stepperSteps.length)
          })
        })
      }), jsx(Content, {
        align: "start",
        side: "bottom",
        css: /*#__PURE__*/css({
          padding: theme.spacing.md
        }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
        children: jsx(Stepper, {
          currentStepIndex: currentStepIndex,
          direction: "vertical",
          localizeStepNumber: localizeStepNumber,
          steps: stepperSteps,
          responsive: false,
          onStepClicked: enableClickingToSteps ? footerProps.goToStep : undefined
        })
      })]
    }), jsxs("div", {
      css: /*#__PURE__*/css({
        display: 'flex',
        flexDirection: 'column',
        columnGap: theme.spacing.lg,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusLg,
        flexGrow: 1,
        padding: padding !== null && padding !== void 0 ? padding : theme.spacing.lg,
        height: isCompact ? `calc(100% - ${ExtraCompactButtonHeight}px)` : '100%',
        maxWidth: MaxWizardContentWidth
      }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
      children: [jsx("div", {
        css: /*#__PURE__*/css({
          flexGrow: expandContentToFullHeight ? 1 : undefined,
          overflowY: disableDefaultScrollBehavior ? 'hidden' : 'auto',
          ...(!disableDefaultScrollBehavior ? getShadowScrollStyles(theme) : {}),
          borderRadius: theme.borders.borderRadiusLg
        }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
        children: wizardSteps[currentStepIndex].content
      }), jsx("div", {
        css: /*#__PURE__*/css({
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'flex-end',
          columnGap: theme.spacing.sm,
          ...(padding !== undefined && {
            padding: theme.spacing.lg
          }),
          paddingTop: theme.spacing.md
        }, process.env.NODE_ENV === "production" ? "" : ";label:VerticalWizardContent;"),
        children: getWizardFooterButtons({
          currentStepIndex: currentStepIndex,
          ...wizardSteps[currentStepIndex],
          ...footerProps,
          moveCancelToOtherSide: true
        })
      })]
    })]
  });
}

function useWizardCurrentStep(_ref) {
  let {
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps,
    onValidateStepChange,
    onStepChanged
  } = _ref;
  const [busyValidatingNextStep, setBusyValidatingNextStep] = useState(false);
  const isLastStep = useMemo(() => currentStepIndex === totalSteps - 1, [currentStepIndex, totalSteps]);
  const onStepsChange = useCallback(async function (step) {
    let completed = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    if (!completed && step === currentStepIndex) return;
    setCurrentStepIndex(step);
    onStepChanged({
      step,
      completed
    });
  }, [currentStepIndex, onStepChanged, setCurrentStepIndex]);
  const goToNextStepOrDone = useCallback(async () => {
    if (onValidateStepChange) {
      setBusyValidatingNextStep(true);
      try {
        const approvedStepChange = await onValidateStepChange(currentStepIndex);
        if (!approvedStepChange) {
          return;
        }
      } finally {
        setBusyValidatingNextStep(false);
      }
    }
    onStepsChange(Math.min(currentStepIndex + 1, totalSteps - 1), isLastStep);
  }, [currentStepIndex, isLastStep, onStepsChange, onValidateStepChange, totalSteps]);
  const goToPreviousStep = useCallback(() => {
    if (currentStepIndex > 0) {
      onStepsChange(currentStepIndex - 1);
    }
  }, [currentStepIndex, onStepsChange]);
  const goToStep = useCallback(step => {
    if (step > -1 && step < totalSteps) {
      onStepsChange(step);
    }
  }, [onStepsChange, totalSteps]);
  return {
    busyValidatingNextStep,
    isLastStep,
    onStepsChange,
    goToNextStepOrDone,
    goToPreviousStep,
    goToStep
  };
}

function Wizard(_ref) {
  let {
    steps,
    onStepChanged,
    onValidateStepChange,
    initialStep,
    ...props
  } = _ref;
  const [currentStepIndex, setCurrentStepIndex] = useState(initialStep !== null && initialStep !== void 0 ? initialStep : 0);
  const currentStepProps = useWizardCurrentStep({
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps: steps.length,
    onStepChanged,
    onValidateStepChange
  });
  return jsx(WizardControlled, {
    ...currentStepProps,
    currentStepIndex: currentStepIndex,
    initialStep: initialStep,
    steps: steps,
    onStepChanged: onStepChanged,
    ...props
  });
}
function WizardControlled(_ref2) {
  let {
    initialStep = 0,
    layout = 'horizontal',
    width = '100%',
    height = '100%',
    steps,
    title,
    ...restOfProps
  } = _ref2;
  if (steps.length === 0 || !_isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
    return null;
  }
  if (layout === 'vertical') {
    return jsx(VerticalWizardContent, {
      width: width,
      height: height,
      steps: steps,
      title: title,
      ...restOfProps
    });
  }
  return jsx(HorizontalWizardContent, {
    width: width,
    height: height,
    steps: steps,
    ...restOfProps
  });
}

function WizardModal(_ref) {
  let {
    onStepChanged,
    onCancel,
    initialStep,
    steps,
    onModalClose,
    localizeStepNumber,
    cancelButtonContent,
    nextButtonContent,
    previousButtonContent,
    doneButtonContent,
    enableClickingToSteps,
    ...modalProps
  } = _ref;
  const [currentStepIndex, setCurrentStepIndex] = useState(initialStep !== null && initialStep !== void 0 ? initialStep : 0);
  const {
    onStepsChange,
    isLastStep,
    ...footerActions
  } = useWizardCurrentStep({
    currentStepIndex,
    setCurrentStepIndex,
    totalSteps: steps.length,
    onStepChanged
  });
  if (steps.length === 0 || !_isUndefined(initialStep) && (initialStep < 0 || initialStep >= steps.length)) {
    return null;
  }
  const footerButtons = getWizardFooterButtons({
    onCancel,
    isLastStep,
    currentStepIndex,
    doneButtonContent,
    previousButtonContent,
    nextButtonContent,
    cancelButtonContent,
    ...footerActions,
    ...steps[currentStepIndex]
  });
  return jsx(Modal, {
    ...modalProps,
    onCancel: onModalClose,
    size: "wide",
    footer: footerButtons,
    children: jsx(HorizontalWizardStepsContent, {
      steps: steps,
      currentStepIndex: currentStepIndex,
      localizeStepNumber: localizeStepNumber,
      enableClickingToSteps: Boolean(enableClickingToSteps),
      goToStep: footerActions.goToStep
    })
  });
}

const foo = 123;

export { Wizard, WizardControlled, WizardModal, foo, useWizardCurrentStep };
//# sourceMappingURL=patterns.js.map
